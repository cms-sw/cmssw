
/*
 *  See header file for a description of this class.
 *
 *  \authors G. Mila , G. Cerminara - INFN Torino
 */

#include "DTNoiseTask.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Geometry
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

// Digi
#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"

// Database
#include "CondFormats/DTObjects/interface/DTTtrig.h"

#include <sstream>
#include <string>

using namespace edm;
using namespace std;

DTNoiseTask::DTNoiseTask(const ParameterSet& ps)
    : evtNumber(0),
      muonGeomToken_(esConsumes<edm::Transition::BeginRun>()),
      tTrigMapToken_(esConsumes<edm::Transition::BeginRun>()) {
  LogVerbatim("DTNoiseTask") << "[DTNoiseTask]: Constructor" << endl;

  //switch for timeBox booking
  doTimeBoxHistos = ps.getUntrackedParameter<bool>("doTbHistos", false);

  // The label to retrieve the digis
  dtDigiToken_ = consumes<DTDigiCollection>(ps.getUntrackedParameter<InputTag>("dtDigiLabel"));

  // the name of the 4D rec hits collection
  recHits4DToken_ =
      consumes<DTRecSegment4DCollection>(edm::InputTag(ps.getUntrackedParameter<string>("recHits4DLabel")));

  // switch for segment veto
  doSegmentVeto = ps.getUntrackedParameter<bool>("doSegmentVeto", false);

  // safe margin (ns) between ttrig and beginning of counting area
  safeMargin = ps.getUntrackedParameter<double>("safeMargin", 200.);
}

DTNoiseTask::~DTNoiseTask() {}

/// Analyze
void DTNoiseTask::analyze(const edm::Event& e, const edm::EventSetup& c) {
  evtNumber++;
  nEventMonitor->Fill(evtNumber);

  if (evtNumber % 1000 == 0)
    LogVerbatim("DTNoiseTask") << "[DTNoiseTask]: Analyzing evt number :" << evtNumber << endl;

  // map of the chambers with at least 1 segment
  std::map<DTChamberId, int> segmentsChId;

  // Get the 4D segment collection from the event
  edm::Handle<DTRecSegment4DCollection> all4DSegments;
  if (doSegmentVeto) {
    e.getByToken(recHits4DToken_, all4DSegments);

    // Loop over all chambers containing a segment and look for the number of segments
    DTRecSegment4DCollection::id_iterator chamberId;
    for (chamberId = all4DSegments->id_begin(); chamberId != all4DSegments->id_end(); ++chamberId) {
      segmentsChId[*chamberId] = 1;
    }
  }

  // Get the digis from the event
  edm::Handle<DTDigiCollection> dtdigis;
  e.getByToken(dtDigiToken_, dtdigis);

  // LOOP OVER ALL THE DIGIS OF THE EVENT
  DTDigiCollection::DigiRangeIterator dtLayerId_It;
  for (dtLayerId_It = dtdigis->begin(); dtLayerId_It != dtdigis->end(); ++dtLayerId_It) {
    for (DTDigiCollection::const_iterator digiIt = ((*dtLayerId_It).second).first;
         digiIt != ((*dtLayerId_It).second).second;
         ++digiIt) {
      //Check the TDC trigger width
      int tdcTime = (*digiIt).countsTDC();
      double upperLimit = tTrigStMap[(*dtLayerId_It).first.superlayerId().chamberId()] - safeMargin;
      if (doTimeBoxHistos)
        tbHistos[(*dtLayerId_It).first.superlayerId()]->Fill(tdcTime);
      if (tdcTime > upperLimit)
        continue;

      //Check the chamber has no 4D segments (optional)
      if (doSegmentVeto && segmentsChId.find((*dtLayerId_It).first.superlayerId().chamberId()) != segmentsChId.end())
        continue;

      // fill the occupancy histo
      // FIXME: needs to be optimized: no need to rescale the histo for each digi
      TH2F* noise_root = noiseHistos[(*dtLayerId_It).first.superlayerId().chamberId()]->getTH2F();
      double normalization = 0;
      if (mapEvt.find((*dtLayerId_It).first.superlayerId().chamberId()) != mapEvt.end()) {
        LogVerbatim("DTNoiseTask") << " Last fill: # of events: "
                                   << mapEvt[(*dtLayerId_It).first.superlayerId().chamberId()] << endl;
        normalization = 1e-9 * upperLimit * mapEvt[(*dtLayerId_It).first.superlayerId().chamberId()];
        // revert back to # of entries
        noise_root->Scale(normalization);
      }
      int yBin = (*dtLayerId_It).first.layer() + (4 * ((*dtLayerId_It).first.superlayerId().superlayer() - 1));
      noise_root->Fill((*digiIt).wire(), yBin);
      // normalize the occupancy histo
      mapEvt[(*dtLayerId_It).first.superlayerId().chamberId()] = evtNumber;
      LogVerbatim("DTNoiseTask") << (*dtLayerId_It).first << " wire: " << (*digiIt).wire()
                                 << " # counts: " << noise_root->GetBinContent((*digiIt).wire(), yBin)
                                 << " Time interval: " << upperLimit << " # of events: " << evtNumber << endl;
      ;
      normalization = double(1e-9 * upperLimit * mapEvt[(*dtLayerId_It).first.superlayerId().chamberId()]);
      // update the rate
      noise_root->Scale(1. / normalization);
      LogVerbatim("DTNoiseTask") << "    noise rate: " << noise_root->GetBinContent((*digiIt).wire(), yBin) << endl;
    }
  }
}

void DTNoiseTask::bookHistos(DQMStore::IBooker& ibooker, DTChamberId chId) {
  // set the folder
  stringstream wheel;
  wheel << chId.wheel();
  stringstream station;
  station << chId.station();
  stringstream sector;
  sector << chId.sector();

  ibooker.setCurrentFolder("DT/05-Noise/Wheel" + wheel.str() +
                           // 			"/Station" + station.str() +
                           "/Sector" + sector.str());

  // Build the histo name
  string histoName = string("NoiseRate") + "_W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str();

  LogVerbatim("DTNoiseTask") << "[DTNoiseTask]: booking chamber histo:" << endl;
  LogVerbatim("DTNoiseTask") << "              folder "
                             << "DT/05-Noise/Wheel" + wheel.str() +
                                    //     "/Station" + station.str() +
                                    "/Sector" + sector.str() + "/"
                             << endl;
  LogVerbatim("DTNoiseTask") << "              histoName " << histoName << endl;

  // Get the chamber from the geometry
  int nWires_max = 0;
  const DTChamber* dtchamber = dtGeom->chamber(chId);
  const vector<const DTSuperLayer*>& superlayers = dtchamber->superLayers();

  // Loop over layers and find the max # of wires
  for (vector<const DTSuperLayer*>::const_iterator sl = superlayers.begin(); sl != superlayers.end();
       ++sl) {  // loop over SLs
    vector<const DTLayer*> layers = (*sl)->layers();
    for (vector<const DTLayer*>::const_iterator lay = layers.begin(); lay != layers.end(); ++lay) {  // loop over layers
      int nWires = (*lay)->specificTopology().channels();
      if (nWires > nWires_max)
        nWires_max = nWires;
    }
  }

  noiseHistos[chId] =
      ibooker.book2D(histoName, "Noise rate (Hz) per channel", nWires_max, 1, nWires_max + 1, 12, 1, 13);
  noiseHistos[chId]->setAxisTitle("wire number", 1);
  noiseHistos[chId]->setBinLabel(1, "SL1-L1", 2);
  noiseHistos[chId]->setBinLabel(2, "SL1-L2", 2);
  noiseHistos[chId]->setBinLabel(3, "SL1-L3", 2);
  noiseHistos[chId]->setBinLabel(4, "SL1-L4", 2);
  noiseHistos[chId]->setBinLabel(5, "SL2-L1", 2);
  noiseHistos[chId]->setBinLabel(6, "SL2-L2", 2);
  noiseHistos[chId]->setBinLabel(7, "SL2-L3", 2);
  noiseHistos[chId]->setBinLabel(8, "SL2-L4", 2);
  noiseHistos[chId]->setBinLabel(9, "SL3-L1", 2);
  noiseHistos[chId]->setBinLabel(10, "SL3-L2", 2);
  noiseHistos[chId]->setBinLabel(11, "SL3-L3", 2);
  noiseHistos[chId]->setBinLabel(12, "SL3-L4", 2);
}

void DTNoiseTask::bookHistos(DQMStore::IBooker& ibooker, DTSuperLayerId slId) {
  // set the folder
  stringstream wheel;
  wheel << slId.chamberId().wheel();
  stringstream station;
  station << slId.chamberId().station();
  stringstream sector;
  sector << slId.chamberId().sector();
  stringstream superlayer;
  superlayer << slId.superlayer();

  ibooker.setCurrentFolder("DT/05-Noise/Wheel" + wheel.str() + "/Station" + station.str() + "/Sector" + sector.str());

  // Build the histo name
  string histoName =
      string("TimeBox") + "_W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str() + "_SL" + superlayer.str();

  LogVerbatim("DTNoiseTask") << "[DTNoiseTask]: booking SL histo:" << endl;
  LogVerbatim("DTNoiseTask") << "              folder "
                             << "DT/05-Noise/Wheel" + wheel.str() + "/Station" + station.str() + "/Sector" +
                                    sector.str() + "/"
                             << endl;
  LogVerbatim("DTNoiseTask") << "              histoName " << histoName << endl;

  tbHistos[slId] = ibooker.book1D(histoName, "Time Box (TDC counts)", 1000, 0, 6000);
}

void DTNoiseTask::dqmBeginRun(const Run& run, const EventSetup& setup) {
  LogVerbatim("DTNoiseTask") << "[DTNoiseTask]: Begin of run" << endl;

  // tTrig Map
  tTrigMap = &setup.getData(tTrigMapToken_);

  // get the geometry
  dtGeom = &setup.getData(muonGeomToken_);
}

void DTNoiseTask::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& run, edm::EventSetup const& setup) {
  ibooker.setCurrentFolder("DT/EventInfo/Counters");
  nEventMonitor = ibooker.bookFloat("nProcessedEventsNoise");

  // Loop over all the chambers
  vector<const DTChamber*>::const_iterator ch_it = dtGeom->chambers().begin();
  vector<const DTChamber*>::const_iterator ch_end = dtGeom->chambers().end();
  for (; ch_it != ch_end; ++ch_it) {
    DTChamberId chId = (*ch_it)->id();
    // histo booking
    bookHistos(ibooker, chId);
    vector<const DTSuperLayer*>::const_iterator sl_it = (*ch_it)->superLayers().begin();
    vector<const DTSuperLayer*>::const_iterator sl_end = (*ch_it)->superLayers().end();
    // Loop over the SLs
    for (; sl_it != sl_end; ++sl_it) {
      DTSuperLayerId slId = (*sl_it)->id();
      if (doTimeBoxHistos)
        bookHistos(ibooker, slId);
      float tTrig, tTrigRMS, kFactor;
      tTrigMap->get(slId, tTrig, tTrigRMS, kFactor, DTTimeUnits::ns);
      // tTrig mapping per station
      // check that the ttrig is the lowest of the 3 SLs
      if (tTrigStMap.find(chId) == tTrigStMap.end() ||
          (tTrigStMap.find(chId) != tTrigStMap.end() && tTrig < tTrigStMap[chId]))
        tTrigStMap[chId] = tTrig;
    }
  }
}

void DTNoiseTask::endLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& setup) {
  LogVerbatim("DTNoiseTask") << "[DTNoiseTask]: End LS, update rates in all histos" << endl;

  // update the rate of all histos (usefull for histos with few entries: they are not updated very often
  for (map<DTChamberId, MonitorElement*>::const_iterator meAndChamber = noiseHistos.begin();
       meAndChamber != noiseHistos.end();
       ++meAndChamber) {
    DTChamberId chId = (*meAndChamber).first;
    TH2F* noise_root = (*meAndChamber).second->getTH2F();
    double upperLimit = tTrigStMap[chId] - safeMargin;

    double normalization = 0;
    if (mapEvt.find(chId) != mapEvt.end()) {
      LogVerbatim("DTNoiseTask") << " Ch: " << chId << " Last fill: # of events: " << mapEvt[chId] << endl;
      normalization = 1e-9 * upperLimit * mapEvt[chId];
      // revert back to # of entries
      noise_root->Scale(normalization);
    }
    //check that event analyzed != 0 might happen oline
    if (evtNumber) {
      // set the # of events analyzed until this update
      LogVerbatim("DTNoiseTask") << "          Update for events: " << evtNumber << endl;
      mapEvt[chId] = evtNumber;
      // update the rate
      normalization = double(1e-9 * upperLimit * evtNumber);
      noise_root->Scale(1. / normalization);
    }
  }
}

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
