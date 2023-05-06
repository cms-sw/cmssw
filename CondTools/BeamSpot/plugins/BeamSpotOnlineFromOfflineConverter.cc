// -*- C++ -*-
//
// Package:    CondTools/BeamSpot
// Class:      BeamSpotOnlineFromOfflineConverter
//
/**\class BeamSpotOnlineFromOfflineConverter BeamSpotOnlineFromOfflineConverter.cc CondTools/BeamSpot/plugins/BeamSpotOnlineFromOfflineConverter.cc

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

class BeamSpotOnlineFromOfflineConverter : public edm::one::EDAnalyzer<> {
public:
  explicit BeamSpotOnlineFromOfflineConverter(const edm::ParameterSet&);
  ~BeamSpotOnlineFromOfflineConverter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  cond::Time_t pack(uint32_t, uint32_t);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  const edm::ESGetToken<BeamSpotObjects, BeamSpotObjectsRcd> bsToken_;

  // IoV-structure
  bool fIsHLT_;
  uint32_t fIOVStartRun_;
  uint32_t fIOVStartLumi_;
  cond::Time_t fnewSince_;
  bool fuseNewSince_;

  // parameters that can't be copied from the BeamSpotObject
  int lastAnalyzedLumi_, lastAnalyzedRun_, lastAnalyzedFill_;
};

//
// constructors and destructor
//
BeamSpotOnlineFromOfflineConverter::BeamSpotOnlineFromOfflineConverter(const edm::ParameterSet& iConfig)
    : bsToken_(esConsumes()) {
  lastAnalyzedLumi_ = iConfig.getParameter<double>("lastAnalyzedLumi");
  lastAnalyzedRun_ = iConfig.getParameter<double>("lastAnalyzedRun");
  lastAnalyzedFill_ = iConfig.getParameter<double>("lastAnalyzedFill");

  fIsHLT_ = iConfig.getParameter<bool>("isHLT");
  if (iConfig.exists("IOVStartRun") && iConfig.exists("IOVStartLumi")) {
    fIOVStartRun_ = iConfig.getUntrackedParameter<uint32_t>("IOVStartRun");
    fIOVStartLumi_ = iConfig.getUntrackedParameter<uint32_t>("IOVStartLumi");
    fnewSince_ = BeamSpotOnlineFromOfflineConverter::pack(fIOVStartRun_, fIOVStartLumi_);
    fuseNewSince_ = true;
    edm::LogPrint("BeamSpotOnlineFromOfflineConverter") << "useNewSince = True";
  } else {
    fuseNewSince_ = false;
    edm::LogPrint("BeamSpotOnlineFromOfflineConverter") << "useNewSince = False";
  }
}

//
// member functions
//

// ------------ Create a since object (cond::Time_t) by packing Run and LS (both uint32_t)  ------------
cond::Time_t BeamSpotOnlineFromOfflineConverter::pack(uint32_t fIOVStartRun, uint32_t fIOVStartLumi) {
  return ((uint64_t)fIOVStartRun << 32 | fIOVStartLumi);
}

// ------------ method called for each event  ------------
void BeamSpotOnlineFromOfflineConverter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const std::string fLabel = (fIsHLT_) ? "BeamSpotOnlineHLTObjectsRcd" : "BeamSpotOnlineLegacyObjectsRcd";
  const BeamSpotObjects* inputSpot = &iSetup.getData(bsToken_);

  BeamSpotOnlineObjects abeam;

  abeam.setLastAnalyzedLumi(lastAnalyzedLumi_);
  abeam.setLastAnalyzedRun(lastAnalyzedRun_);
  abeam.setLastAnalyzedFill(lastAnalyzedFill_);
  abeam.setStartTimeStamp(std::time(nullptr));
  abeam.setEndTimeStamp(std::time(nullptr));
  abeam.setType(inputSpot->beamType());
  abeam.setPosition(inputSpot->x(), inputSpot->y(), inputSpot->z());
  abeam.setSigmaZ(inputSpot->sigmaZ());
  abeam.setdxdz(inputSpot->dxdz());
  abeam.setdydz(inputSpot->dydz());
  abeam.setBeamWidthX(inputSpot->beamWidthX());
  abeam.setBeamWidthY(inputSpot->beamWidthY());
  abeam.setEmittanceX(inputSpot->emittanceX());
  abeam.setEmittanceY(inputSpot->emittanceY());
  abeam.setBetaStar(inputSpot->betaStar());

  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 7; ++j) {
      abeam.setCovariance(i, j, inputSpot->covariance(i, j));
    }
  }

  // Set the creation time of the payload to the current time
  auto creationTime =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  abeam.setCreationTime(creationTime);

  edm::LogPrint("BeamSpotOnlineFromOfflineConverter") << " Writing results to DB...";

  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable()) {
    edm::LogPrint("BeamSpotOnlineFromOfflineConverter") << "poolDBService available";
    if (poolDbService->isNewTagRequest(fLabel)) {
      edm::LogPrint("BeamSpotOnlineFromOfflineConverter") << "new tag requested";
      if (fuseNewSince_) {
        edm::LogPrint("BeamSpotOnlineFromOfflineConverter") << "Using a new Since: " << fnewSince_;
        poolDbService->createOneIOV<BeamSpotOnlineObjects>(abeam, fnewSince_, fLabel);
      } else
        poolDbService->createOneIOV<BeamSpotOnlineObjects>(abeam, poolDbService->beginOfTime(), fLabel);
    } else {
      edm::LogPrint("BeamSpotOnlineFromOfflineConverter") << "no new tag requested";
      if (fuseNewSince_) {
        edm::LogPrint("BeamSpotOnlineFromOfflineConverter") << "Using a new Since: " << fnewSince_;
        poolDbService->appendOneIOV<BeamSpotOnlineObjects>(abeam, fnewSince_, fLabel);
      } else
        poolDbService->appendOneIOV<BeamSpotOnlineObjects>(abeam, poolDbService->currentTime(), fLabel);
    }
  }
  edm::LogPrint("BeamSpotOnlineFromOfflineConverter") << "[BeamSpotOnlineFromOfflineConverter] endJob done \n";
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void BeamSpotOnlineFromOfflineConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("isHLT", true);
  desc.addOptionalUntracked<uint32_t>("IOVStartRun", 1);
  desc.addOptionalUntracked<uint32_t>("IOVStartLumi", 1);
  desc.add<double>("lastAnalyzedLumi", 1000);
  desc.add<double>("lastAnalyzedRun", 1);
  desc.add<double>("lastAnalyzedFill", -999);
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(BeamSpotOnlineFromOfflineConverter);
