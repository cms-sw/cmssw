// -*- C++ -*-
//
// Package:    CondTools/BeamSpot
// Class:      BeamSpotOnlineShifter
//
/**\class BeamSpotOnlineShifter BeamSpotOnlineShifter.cc CondTools/BeamSpot/plugins/BeamSpotOnlineShifter.cc

 Description: EDAnalyzer to create a BeamSpotOnlineHLTObjectsRcd from a BeamSpotObjectsRcd (inserting some parameters manually)

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Marco Musich
//         Created:  Sat, 06 May 2023 21:10:00 GMT
//
//

// system include files
#include <memory>
#include <string>
#include <fstream>
#include <iostream>
#include <ctime>

// user include files
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotOnlineObjects.h"
#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
#include "CondFormats/DataRecord/interface/BeamSpotOnlineHLTObjectsRcd.h"
#include "CondFormats/DataRecord/interface/BeamSpotOnlineLegacyObjectsRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

//
// class declaration
//

class BeamSpotOnlineShifter : public edm::one::EDAnalyzer<> {
public:
  explicit BeamSpotOnlineShifter(const edm::ParameterSet&);
  ~BeamSpotOnlineShifter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  cond::Time_t pack(uint32_t, uint32_t);
  template <class Record>
  void writeToDB(const edm::Event& iEvent,
                 const edm::EventSetup& iSetup,
                 const edm::ESGetToken<BeamSpotOnlineObjects, Record>& token);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------

  edm::ESGetToken<BeamSpotOnlineObjects, BeamSpotOnlineHLTObjectsRcd> hltToken;
  edm::ESGetToken<BeamSpotOnlineObjects, BeamSpotOnlineLegacyObjectsRcd> legacyToken;

  edm::ESWatcher<BeamSpotOnlineHLTObjectsRcd> bsHLTWatcher_;
  edm::ESWatcher<BeamSpotOnlineLegacyObjectsRcd> bsLegayWatcher_;

  // IoV-structure
  const bool fIsHLT_;
  const double xShift_;
  const double yShift_;
  const double zShift_;
  uint32_t fIOVStartRun_;
  uint32_t fIOVStartLumi_;
  cond::Time_t fnewSince_;
  bool fuseNewSince_;
  std::string fLabel_;
};

//
// constructors and destructor
//
BeamSpotOnlineShifter::BeamSpotOnlineShifter(const edm::ParameterSet& iConfig)
    : fIsHLT_(iConfig.getParameter<bool>("isHLT")),
      xShift_(iConfig.getParameter<double>("xShift")),
      yShift_(iConfig.getParameter<double>("yShift")),
      zShift_(iConfig.getParameter<double>("zShift")) {
  if (iConfig.exists("IOVStartRun") && iConfig.exists("IOVStartLumi")) {
    fIOVStartRun_ = iConfig.getUntrackedParameter<uint32_t>("IOVStartRun");
    fIOVStartLumi_ = iConfig.getUntrackedParameter<uint32_t>("IOVStartLumi");
    fnewSince_ = BeamSpotOnlineShifter::pack(fIOVStartRun_, fIOVStartLumi_);
    fuseNewSince_ = true;
    edm::LogPrint("BeamSpotOnlineShifter") << "useNewSince = True";
  } else {
    fuseNewSince_ = false;
    edm::LogPrint("BeamSpotOnlineShifter") << "useNewSince = False";
  }
  fLabel_ = (fIsHLT_) ? "BeamSpotOnlineHLTObjectsRcd" : "BeamSpotOnlineLegacyObjectsRcd";

  if (fIsHLT_) {
    hltToken = esConsumes();
  } else {
    legacyToken = esConsumes();
  }
}

//
// member functions
//

// ------------ Create a since object (cond::Time_t) by packing Run and LS (both uint32_t)  ------------
cond::Time_t BeamSpotOnlineShifter::pack(uint32_t fIOVStartRun, uint32_t fIOVStartLumi) {
  return ((uint64_t)fIOVStartRun << 32 | fIOVStartLumi);
}

template <class Record>
void BeamSpotOnlineShifter::writeToDB(const edm::Event& iEvent,
                                      const edm::EventSetup& iSetup,
                                      const edm::ESGetToken<BeamSpotOnlineObjects, Record>& token) {
  // input object
  const BeamSpotOnlineObjects* inputSpot = &iSetup.getData(token);

  // output object
  BeamSpotOnlineObjects abeam;

  abeam.setPosition(inputSpot->x() + xShift_, inputSpot->y() + yShift_, inputSpot->z() + zShift_);
  abeam.setSigmaZ(inputSpot->sigmaZ());
  abeam.setdxdz(inputSpot->dxdz());
  abeam.setdydz(inputSpot->dydz());
  abeam.setBeamWidthX(inputSpot->beamWidthX());
  abeam.setBeamWidthY(inputSpot->beamWidthY());
  abeam.setBeamWidthXError(inputSpot->beamWidthXError());
  abeam.setBeamWidthYError(inputSpot->beamWidthYError());

  for (unsigned int i = 0; i < 7; i++) {
    for (unsigned int j = 0; j < 7; j++) {
      abeam.setCovariance(i, j, inputSpot->covariance(i, j));
    }
  }

  abeam.setType(inputSpot->beamType());
  abeam.setEmittanceX(inputSpot->emittanceX());
  abeam.setEmittanceY(inputSpot->emittanceY());
  abeam.setBetaStar(inputSpot->betaStar());

  // online BeamSpot object specific
  abeam.setLastAnalyzedLumi(inputSpot->lastAnalyzedLumi());
  abeam.setLastAnalyzedRun(inputSpot->lastAnalyzedRun());
  abeam.setLastAnalyzedFill(inputSpot->lastAnalyzedFill());
  abeam.setStartTimeStamp(inputSpot->startTimeStamp());
  abeam.setEndTimeStamp(inputSpot->endTimeStamp());
  abeam.setNumTracks(inputSpot->numTracks());
  abeam.setNumPVs(inputSpot->numPVs());
  abeam.setUsedEvents(inputSpot->usedEvents());
  abeam.setMaxPVs(inputSpot->maxPVs());
  abeam.setMeanPV(inputSpot->meanPV());
  abeam.setMeanErrorPV(inputSpot->meanErrorPV());
  abeam.setRmsPV(inputSpot->rmsPV());
  abeam.setRmsErrorPV(inputSpot->rmsErrorPV());
  abeam.setStartTime(inputSpot->startTime());
  abeam.setEndTime(inputSpot->endTime());
  abeam.setLumiRange(inputSpot->lumiRange());
  abeam.setCreationTime(inputSpot->creationTime());

  edm::LogPrint("BeamSpotOnlineShifter") << " Writing results to DB...";
  edm::LogPrint("BeamSpotOnlineShifter") << abeam;

  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable()) {
    edm::LogPrint("BeamSpotOnlineShifter") << "poolDBService available";
    if (poolDbService->isNewTagRequest(fLabel_)) {
      edm::LogPrint("BeamSpotOnlineShifter") << "new tag requested";
      if (fuseNewSince_) {
        edm::LogPrint("BeamSpotOnlineShifter") << "Using a new Since: " << fnewSince_;
        poolDbService->createOneIOV<BeamSpotOnlineObjects>(abeam, fnewSince_, fLabel_);
      } else
        poolDbService->createOneIOV<BeamSpotOnlineObjects>(abeam, poolDbService->beginOfTime(), fLabel_);
    } else {
      edm::LogPrint("BeamSpotOnlineShifter") << "no new tag requested";
      if (fuseNewSince_) {
        cond::Time_t thisSince = BeamSpotOnlineShifter::pack(iEvent.getLuminosityBlock().run(),
                                                             iEvent.getLuminosityBlock().luminosityBlock());
        edm::LogPrint("BeamSpotOnlineShifter") << "Using a new Since: " << thisSince;
        poolDbService->appendOneIOV<BeamSpotOnlineObjects>(abeam, thisSince, fLabel_);
      } else
        poolDbService->appendOneIOV<BeamSpotOnlineObjects>(abeam, poolDbService->currentTime(), fLabel_);
    }
  }
  edm::LogPrint("BeamSpotOnlineShifter") << "[BeamSpotOnlineShifter] analyze done \n";
}

// ------------ method called for each event  ------------
void BeamSpotOnlineShifter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  if (fIsHLT_) {
    if (bsHLTWatcher_.check(iSetup)) {
      writeToDB<BeamSpotOnlineHLTObjectsRcd>(iEvent, iSetup, hltToken);
    }
  } else {
    if (bsLegayWatcher_.check(iSetup)) {
      writeToDB<BeamSpotOnlineLegacyObjectsRcd>(iEvent, iSetup, legacyToken);
    }
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void BeamSpotOnlineShifter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("isHLT", true);
  desc.add<double>("xShift", 0.0)->setComment("in cm");
  desc.add<double>("yShift", 0.0)->setComment("in cm");
  desc.add<double>("zShift", 0.0)->setComment("in cm");
  desc.addOptionalUntracked<uint32_t>("IOVStartRun", 1);
  desc.addOptionalUntracked<uint32_t>("IOVStartLumi", 1);
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(BeamSpotOnlineShifter);
