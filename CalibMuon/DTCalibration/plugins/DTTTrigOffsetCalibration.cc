
/*
 *  See header file for a description of this class.
 *
 *  \author A. Vilela Pereira
 */

#include "DTTTrigOffsetCalibration.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"
#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"

#include "CalibMuon/DTCalibration/interface/DTCalibDBUtils.h"
#include "CalibMuon/DTCalibration/interface/DTSegmentSelector.h"

#include <string>
#include <sstream>
#include "TFile.h"
#include "TH1F.h"

using namespace std;
using namespace edm;

DTTTrigOffsetCalibration::DTTTrigOffsetCalibration(const ParameterSet& pset)
    : theRecHits4DToken_(consumes<DTRecSegment4DCollection>(pset.getParameter<InputTag>("recHits4DLabel"))),
      doTTrigCorrection_(pset.getUntrackedParameter<bool>("doT0SegCorrection", false)),
      theCalibChamber_(pset.getUntrackedParameter<string>("calibChamber", "All")),
      ttrigToken_(
          esConsumes<edm::Transition::BeginRun>(edm::ESInputTag("", pset.getUntrackedParameter<string>("dbLabel")))),
      dtGeomToken_(esConsumes()) {
  LogVerbatim("Calibration") << "[DTTTrigOffsetCalibration] Constructor called!";

  edm::ConsumesCollector collector(consumesCollector());
  select_ = new DTSegmentSelector(pset, collector);

  // the root file which will contain the histos
  string rootFileName = pset.getUntrackedParameter<string>("rootFileName", "DTT0SegHistos.root");
  rootFile_ = new TFile(rootFileName.c_str(), "RECREATE");
  rootFile_->cd();
}

void DTTTrigOffsetCalibration::beginRun(const edm::Run& run, const edm::EventSetup& setup) {
  if (doTTrigCorrection_) {
    ESHandle<DTTtrig> tTrig;
    tTrig = setup.getHandle(ttrigToken_);
    tTrigMap_ = &setup.getData(ttrigToken_);
    LogVerbatim("Calibration") << "[DTTTrigOffsetCalibration]: TTrig version: " << tTrig->version() << endl;
  }
}

DTTTrigOffsetCalibration::~DTTTrigOffsetCalibration() {
  rootFile_->Close();
  LogVerbatim("Calibration") << "[DTTTrigOffsetCalibration] Destructor called!";
}

void DTTTrigOffsetCalibration::analyze(const Event& event, const EventSetup& eventSetup) {
  rootFile_->cd();
  DTChamberId chosenChamberId;

  if (theCalibChamber_ != "All") {
    stringstream linestr;
    int selWheel, selStation, selSector;
    linestr << theCalibChamber_;
    linestr >> selWheel >> selStation >> selSector;
    chosenChamberId = DTChamberId(selWheel, selStation, selSector);
    LogVerbatim("Calibration") << " chosen chamber " << chosenChamberId << endl;
  }

  // Get the DT Geometry
  ESHandle<DTGeometry> dtGeom;
  dtGeom = eventSetup.getHandle(dtGeomToken_);

  // Get the rechit collection from the event
  const Handle<DTRecSegment4DCollection>& all4DSegments = event.getHandle(theRecHits4DToken_);

  // Loop over segments by chamber
  DTRecSegment4DCollection::id_iterator chamberIdIt;
  for (chamberIdIt = all4DSegments->id_begin(); chamberIdIt != all4DSegments->id_end(); ++chamberIdIt) {
    // Get the chamber from the setup
    const DTChamber* chamber = dtGeom->chamber(*chamberIdIt);
    LogTrace("Calibration") << "Chamber Id: " << *chamberIdIt;

    // Book histos
    if (theT0SegHistoMap_.find(*chamberIdIt) == theT0SegHistoMap_.end()) {
      bookHistos(*chamberIdIt);
    }

    // Calibrate just the chosen chamber/s
    if ((theCalibChamber_ != "All") && ((*chamberIdIt) != chosenChamberId))
      continue;

    // Get the range for the corresponding ChamberId
    DTRecSegment4DCollection::range range = all4DSegments->get((*chamberIdIt));

    // Loop over the rechits of this DetUnit
    for (DTRecSegment4DCollection::const_iterator segment = range.first; segment != range.second; ++segment) {
      LogTrace("Calibration") << "Segment local pos (in chamber RF): " << (*segment).localPosition()
                              << "\nSegment global pos: " << chamber->toGlobal((*segment).localPosition());

      if (!(*select_)(*segment, event, eventSetup))
        continue;

      // Fill t0-seg values
      if ((*segment).hasPhi()) {
        //if( segment->phiSegment()->ist0Valid() ){
        if ((segment->phiSegment()->t0()) != 0.00) {
          (theT0SegHistoMap_[*chamberIdIt])[0]->Fill(segment->phiSegment()->t0());
        }
      }
      if ((*segment).hasZed()) {
        //if( segment->zSegment()->ist0Valid() ){
        if ((segment->zSegment()->t0()) != 0.00) {
          (theT0SegHistoMap_[*chamberIdIt])[1]->Fill(segment->zSegment()->t0());
        }
      }
    }  // DTRecSegment4DCollection::const_iterator segment
  }    // DTRecSegment4DCollection::id_iterator chamberIdIt
}  // DTTTrigOffsetCalibration::analyze

void DTTTrigOffsetCalibration::endJob() {
  rootFile_->cd();

  LogVerbatim("Calibration") << "[DTTTrigOffsetCalibration] Writing histos to file!" << endl;

  for (ChamberHistosMap::const_iterator itChHistos = theT0SegHistoMap_.begin(); itChHistos != theT0SegHistoMap_.end();
       ++itChHistos) {
    for (vector<TH1F*>::const_iterator itHist = (*itChHistos).second.begin(); itHist != (*itChHistos).second.end();
         ++itHist)
      (*itHist)->Write();
  }

  if (doTTrigCorrection_) {
    // Create the object to be written to DB
    DTTtrig* tTrig = new DTTtrig();

    for (ChamberHistosMap::const_iterator itChHistos = theT0SegHistoMap_.begin(); itChHistos != theT0SegHistoMap_.end();
         ++itChHistos) {
      DTChamberId chId = itChHistos->first;
      // Get SuperLayerId's for each ChamberId
      vector<DTSuperLayerId> slIds;
      slIds.push_back(DTSuperLayerId(chId, 1));
      slIds.push_back(DTSuperLayerId(chId, 3));
      if (chId.station() != 4)
        slIds.push_back(DTSuperLayerId(chId, 2));

      for (vector<DTSuperLayerId>::const_iterator itSl = slIds.begin(); itSl != slIds.end(); ++itSl) {
        // Get old values from DB
        float ttrigMean = 0;
        float ttrigSigma = 0;
        float kFactor = 0;
        tTrigMap_->get(*itSl, ttrigMean, ttrigSigma, kFactor, DTTimeUnits::ns);
        //FIXME: verify if values make sense
        // Set new values
        float ttrigMeanNew = ttrigMean;
        float ttrigSigmaNew = ttrigSigma;
        float t0SegMean =
            (itSl->superLayer() != 2) ? itChHistos->second[0]->GetMean() : itChHistos->second[1]->GetMean();

        float kFactorNew = (kFactor * ttrigSigma + t0SegMean) / ttrigSigma;

        tTrig->set(*itSl, ttrigMeanNew, ttrigSigmaNew, kFactorNew, DTTimeUnits::ns);
      }
    }
    LogVerbatim("Calibration") << "[DTTTrigOffsetCalibration] Writing ttrig object to DB!" << endl;
    // Write the object to DB
    string tTrigRecord = "DTTtrigRcd";
    DTCalibDBUtils::writeToDB(tTrigRecord, tTrig);
  }
}

// Book a set of histograms for a given Chamber
void DTTTrigOffsetCalibration::bookHistos(DTChamberId chId) {
  LogTrace("Calibration") << "   Booking histos for Chamber: " << chId;

  // Compose the chamber name
  std::string wheel = std::to_string(chId.wheel());
  std::string station = std::to_string(chId.station());
  std::string sector = std::to_string(chId.sector());

  string chHistoName = "_W" + wheel + "_St" + station + "_Sec" + sector;

  vector<TH1F*> histos;
  // Note the order matters
  histos.push_back(new TH1F(("hRPhiSegT0" + chHistoName).c_str(), "t0 from Phi segments", 500, -60., 60.));
  if (chId.station() != 4)
    histos.push_back(new TH1F(("hRZSegT0" + chHistoName).c_str(), "t0 from Z segments", 500, -60., 60.));

  theT0SegHistoMap_[chId] = histos;
}
