/*
 * \file DTLocalTriggerSynchTask.cc
 *
 * \author C. Battilana - CIEMAT
 *
*/

#include "DQM/DTMonitorModule/src/DTLocalTriggerSynchTask.h"

// Framework
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"

// Geometry
#include "DataFormats/GeometryVector/interface/Pi.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

// tTrigs
#include "CalibMuon/DTDigiSync/interface/DTTTrigSyncFactory.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigBaseSync.h"

// DT Digi
#include <DataFormats/DTDigi/interface/DTDigi.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>

//Root
#include "TH1.h"
#include "TAxis.h"

#include <sstream>
#include <iostream>
#include <fstream>

using namespace edm;
using namespace std;

DTLocalTriggerSynchTask::DTLocalTriggerSynchTask(const edm::ParameterSet& ps)
    : nevents(0),
      tTrigSync{DTTTrigSyncFactory::get()->create(ps.getParameter<std::string>("tTrigMode"),
                                                  ps.getParameter<edm::ParameterSet>("tTrigModeConfig"))} {
  edm::LogVerbatim("DTLocalTriggerSynchTask") << "[DTLocalTriggerSynchTask]: Constructor" << endl;
  tm_Token_ = consumes<L1MuDTChambPhContainer>(ps.getParameter<edm::InputTag>("TMInputTag"));
  seg_Token_ = consumes<DTRecSegment4DCollection>(ps.getParameter<edm::InputTag>("SEGInputTag"));

  bxTime = ps.getParameter<double>("bxTimeInterval");  // CB move this to static const or DB
  rangeInBX = ps.getParameter<bool>("rangeWithinBX");
  nBXLow = ps.getParameter<int>("nBXLow");
  nBXHigh = ps.getParameter<int>("nBXHigh");
  angleRange = ps.getParameter<double>("angleRange");
  minHitsPhi = ps.getParameter<int>("minHitsPhi");
  baseDirectory = ps.getParameter<string>("baseDir");
}

DTLocalTriggerSynchTask::~DTLocalTriggerSynchTask() {
  edm::LogVerbatim("DTLocalTriggerSynchTask") << "[DTLocalTriggerSynchTask]: analyzed " << nevents << " events" << endl;
}

void DTLocalTriggerSynchTask::bookHistograms(DQMStore::IBooker& ibooker,
                                             edm::Run const& iRun,
                                             edm::EventSetup const& context) {
  edm::LogVerbatim("DTLocalTriggerSynchTask") << "[DTLocalTriggerSynchTask]: Book Histograms" << endl;

  ibooker.setCurrentFolder(baseDir());
  ibooker.bookFloat("BXTimeSpacing")->Fill(bxTime);

  tTrigSync->setES(context);

  std::vector<const DTChamber*>::const_iterator chambIt = muonGeom->chambers().begin();
  std::vector<const DTChamber*>::const_iterator chambEnd = muonGeom->chambers().end();

  for (; chambIt != chambEnd; ++chambIt) {
    bookHistos(ibooker, (*chambIt)->id());
    triggerHistos[(*chambIt)->id().rawId()]["tTrig_SL1"]->Fill(tTrigSync->offset(DTWireId((*chambIt)->id(), 1, 1, 2)));
    triggerHistos[(*chambIt)->id().rawId()]["tTrig_SL3"]->Fill(tTrigSync->offset(DTWireId((*chambIt)->id(), 3, 1, 2)));
  }
}

void DTLocalTriggerSynchTask::dqmBeginRun(const Run& run, const EventSetup& context) {
  context.get<MuonGeometryRecord>().get(muonGeom);
}

void DTLocalTriggerSynchTask::analyze(const edm::Event& event, const edm::EventSetup& context) {
  nevents++;

  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 6; ++j) {
      for (int k = 0; k < 13; ++k) {
        phCodeBestTM[j][i][k] = -1;
      }
    }
  }

  // Get best TM triggers
  edm::Handle<L1MuDTChambPhContainer> l1DTTPGPh;
  event.getByToken(tm_Token_, l1DTTPGPh);
  vector<L1MuDTChambPhDigi> const* phTrigs = l1DTTPGPh->getContainer();

  vector<L1MuDTChambPhDigi>::const_iterator iph = phTrigs->begin();
  vector<L1MuDTChambPhDigi>::const_iterator iphe = phTrigs->end();
  for (; iph != iphe; ++iph) {
    int phwheel = iph->whNum();
    int phsec = iph->scNum() + 1;  // DTTF[0-11] -> DT[1-12] Sector Numbering
    int phst = iph->stNum();
    int phcode = iph->code();

    if (phcode > phCodeBestTM[phwheel + 3][phst][phsec] && phcode < 7) {
      phCodeBestTM[phwheel + 3][phst][phsec] = phcode;
    }
  }

  //Get best segments (highest number of phi hits)
  vector<const DTRecSegment4D*> bestSegments4D;
  Handle<DTRecSegment4DCollection> segments4D;
  event.getByToken(seg_Token_, segments4D);
  DTRecSegment4DCollection::const_iterator track;
  DTRecSegment4DCollection::id_iterator chambIdIt;

  for (chambIdIt = segments4D->id_begin(); chambIdIt != segments4D->id_end(); ++chambIdIt) {
    DTRecSegment4DCollection::range range = segments4D->get(*chambIdIt);
    const DTRecSegment4D* best = nullptr;
    int hitsBest = 0;
    int hits = 0;

    for (track = range.first; track != range.second; ++track) {
      if ((*track).hasPhi()) {
        hits = (*track).phiSegment()->degreesOfFreedom() + 2;
        if (hits > hitsBest) {
          best = &(*track);
          hitsBest = hits;
        }
      }
    }
    if (best) {
      bestSegments4D.push_back(best);
    }
  }

  // Filling histos
  vector<const DTRecSegment4D*>::const_iterator bestSegIt = bestSegments4D.begin();
  vector<const DTRecSegment4D*>::const_iterator bestSegEnd = bestSegments4D.end();
  for (; bestSegIt != bestSegEnd; ++bestSegIt) {
    float dir = atan((*bestSegIt)->localDirection().x() / (*bestSegIt)->localDirection().z()) * 180 /
                Geom::pi();  // CB cerca un modo migliore x farlo
    const DTRecSegment2D* seg2D = (*bestSegIt)->phiSegment();
    int nHitsPhi = seg2D->degreesOfFreedom() + 2;
    DTChamberId chambId = (*bestSegIt)->chamberId();
    map<string, MonitorElement*>& innerME = triggerHistos[chambId.rawId()];

    if (fabs(dir) < angleRange && nHitsPhi >= minHitsPhi && seg2D->ist0Valid()) {
      float t0seg = (*bestSegIt)->phiSegment()->t0();
      float tTrig = (tTrigSync->offset(DTWireId(chambId, 1, 1, 2)) + tTrigSync->offset(DTWireId(chambId, 3, 1, 2))) / 2;
      float time = tTrig + t0seg;
      float htime = rangeInBX ? time - int(time / bxTime) * bxTime : time - int(tTrig / bxTime) * bxTime;

      int wheel = chambId.wheel();
      int sector = chambId.sector();
      int station = chambId.station();
      int scsector = sector > 12 ? sector == 13 ? 4 : 10 : sector;

      int qualTM = phCodeBestTM[wheel + 3][station][scsector];

      if (fabs(t0seg) > 0.01) {
        innerME.find("SEG_TrackCrossingTime")->second->Fill(htime);
        if (qualTM >= 0)
          innerME.find("TM_TrackCrossingTimeAll")->second->Fill(htime);
        if (qualTM == 6)
          innerME.find("TM_TrackCrossingTimeHH")->second->Fill(htime);
      }
    }
  }
}

void DTLocalTriggerSynchTask::bookHistos(DQMStore::IBooker& ibooker, const DTChamberId& dtChId) {
  stringstream wheel;
  wheel << dtChId.wheel();
  stringstream station;
  station << dtChId.station();
  stringstream sector;
  sector << dtChId.sector();
  uint32_t chRawId = dtChId.rawId();

  ibooker.setCurrentFolder(baseDir() + "/Wheel" + wheel.str() + "/Sector" + sector.str() + "/Station" + station.str());

  std::vector<string> histoTags = {"SEG_TrackCrossingTime", "TM_TrackCrossingTimeAll", "TM_TrackCrossingTimeHH"};

  float min = rangeInBX ? 0 : nBXLow * bxTime;
  float max = rangeInBX ? bxTime : nBXHigh * bxTime;
  int nbins = static_cast<int>(ceil(rangeInBX ? bxTime : (nBXHigh - nBXLow) * bxTime));

  for (const auto& histoTag : histoTags) {
    string histoName =
        histoTag + (rangeInBX ? "InBX" : "") + "_W" + wheel.str() + "_Sec" + sector.str() + "_St" + station.str();
    edm::LogVerbatim("DTLocalTriggerSynchTask")
        << "[DTLocalTriggerSynchTask]: booking " << baseDir() + "/Wheel" << wheel.str() << "/Sector" << sector.str()
        << "/Station" << station.str() << "/" << histoName << endl;

    triggerHistos[chRawId][histoTag] = ibooker.book1D(histoName.c_str(), "Track time distribution", nbins, min, max);
  }

  string floatTag[2] = {"tTrig_SL1", "tTrig_SL3"};

  for (int iFloat = 0; iFloat < 2; ++iFloat) {
    string floatName = floatTag[iFloat] + "_W" + wheel.str() + "_Sec" + sector.str() + "_St" + station.str();
    triggerHistos[chRawId][floatTag[iFloat]] = ibooker.bookFloat(floatName);
  }
}

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
