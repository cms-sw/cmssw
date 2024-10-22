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
  edm::ESWatcher<BeamSpotObjectsRcd> bsWatcher_;

  // parameters that can't be copied from the BeamSpotObject
  const int lastAnalyzedLumi_, lastAnalyzedRun_, lastAnalyzedFill_;
  const int numTracks_, numPVs_, numUsedEvents_, numMaxPVs_;
  const float meanPVs_, meanPVError_, rmsPV_, rmsPVError_;
  const std::string startTime_, endTime_, lumiRange_;

  // IoV-structure
  const bool fIsHLT_;
  uint32_t fIOVStartRun_;
  uint32_t fIOVStartLumi_;
  cond::Time_t fnewSince_;
  bool fuseNewSince_;
  std::string fLabel_;
};

//
// constructors and destructor
//
BeamSpotOnlineFromOfflineConverter::BeamSpotOnlineFromOfflineConverter(const edm::ParameterSet& iConfig)
    : bsToken_(esConsumes()),
      lastAnalyzedLumi_(iConfig.getParameter<double>("lastAnalyzedLumi")),
      lastAnalyzedRun_(iConfig.getParameter<double>("lastAnalyzedRun")),
      lastAnalyzedFill_(iConfig.getParameter<double>("lastAnalyzedFill")),
      numTracks_(iConfig.getParameter<int>("numTracks")),
      numPVs_(iConfig.getParameter<int>("numPVs")),
      numUsedEvents_(iConfig.getParameter<int>("numUsedEvents")),
      numMaxPVs_(iConfig.getParameter<int>("numMaxPVs")),
      meanPVs_(iConfig.getParameter<double>("meanPVs")),
      meanPVError_(iConfig.getParameter<double>("meanPVError")),
      rmsPV_(iConfig.getParameter<double>("rmsPVs")),
      rmsPVError_(iConfig.getParameter<double>("rmsPVError")),
      startTime_(iConfig.getParameter<std::string>("startTime")),
      endTime_(iConfig.getParameter<std::string>("endTime")),
      lumiRange_(iConfig.getParameter<std::string>("lumiRange")),
      fIsHLT_(iConfig.getParameter<bool>("isHLT")) {
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
  fLabel_ = (fIsHLT_) ? "BeamSpotOnlineHLTObjectsRcd" : "BeamSpotOnlineLegacyObjectsRcd";
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
  if (bsWatcher_.check(iSetup)) {
    const BeamSpotObjects* inputSpot = &iSetup.getData(bsToken_);

    BeamSpotOnlineObjects abeam;
    abeam.copyFromBeamSpotObject(*inputSpot);
    abeam.setLastAnalyzedLumi(lastAnalyzedLumi_);
    abeam.setLastAnalyzedRun(lastAnalyzedRun_);
    abeam.setLastAnalyzedFill(lastAnalyzedFill_);
    abeam.setStartTimeStamp(std::time(nullptr));
    abeam.setEndTimeStamp(std::time(nullptr));
    abeam.setNumTracks(numTracks_);
    abeam.setNumPVs(numPVs_);
    abeam.setUsedEvents(numUsedEvents_);
    abeam.setMaxPVs(numMaxPVs_);
    abeam.setMeanPV(meanPVs_);
    abeam.setMeanErrorPV(meanPVError_);
    abeam.setRmsPV(rmsPV_);
    abeam.setRmsErrorPV(rmsPVError_);
    abeam.setStartTime(startTime_);
    abeam.setEndTime(endTime_);
    abeam.setLumiRange(lumiRange_);

    // Set the creation time of the payload to the current time
    auto creationTime =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();
    abeam.setCreationTime(creationTime);

    edm::LogPrint("BeamSpotOnlineFromOfflineConverter") << " Writing results to DB...";
    edm::LogPrint("BeamSpotOnlineFromOfflineConverter") << abeam;

    edm::Service<cond::service::PoolDBOutputService> poolDbService;
    if (poolDbService.isAvailable()) {
      edm::LogPrint("BeamSpotOnlineFromOfflineConverter") << "poolDBService available";
      if (poolDbService->isNewTagRequest(fLabel_)) {
        edm::LogPrint("BeamSpotOnlineFromOfflineConverter") << "new tag requested";
        if (fuseNewSince_) {
          edm::LogPrint("BeamSpotOnlineFromOfflineConverter") << "Using a new Since: " << fnewSince_;
          poolDbService->createOneIOV<BeamSpotOnlineObjects>(abeam, fnewSince_, fLabel_);
        } else
          poolDbService->createOneIOV<BeamSpotOnlineObjects>(abeam, poolDbService->beginOfTime(), fLabel_);
      } else {
        edm::LogPrint("BeamSpotOnlineFromOfflineConverter") << "no new tag requested";
        if (fuseNewSince_) {
          cond::Time_t thisSince = BeamSpotOnlineFromOfflineConverter::pack(
              iEvent.getLuminosityBlock().run(), iEvent.getLuminosityBlock().luminosityBlock());
          edm::LogPrint("BeamSpotOnlineFromOfflineConverter") << "Using a new Since: " << thisSince;
          poolDbService->appendOneIOV<BeamSpotOnlineObjects>(abeam, thisSince, fLabel_);
        } else
          poolDbService->appendOneIOV<BeamSpotOnlineObjects>(abeam, poolDbService->currentTime(), fLabel_);
      }
    }
    edm::LogPrint("BeamSpotOnlineFromOfflineConverter") << "[BeamSpotOnlineFromOfflineConverter] analyze done \n";
  }
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
  desc.add<int>("numTracks", 0);
  desc.add<int>("numPVs", 0);
  desc.add<int>("numUsedEvents", 0);
  desc.add<int>("numMaxPVs", 0);
  desc.add<double>("meanPVs", 0.);
  desc.add<double>("meanPVError", 0.);
  desc.add<double>("rmsPVs", 0.);
  desc.add<double>("rmsPVError", 0.);
  desc.add<std::string>("startTime", std::string(""));
  desc.add<std::string>("endTime", std::string(""));
  desc.add<std::string>("lumiRange", std::string(""));
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(BeamSpotOnlineFromOfflineConverter);
