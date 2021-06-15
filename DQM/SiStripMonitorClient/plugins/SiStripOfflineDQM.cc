// -*- C++ -*-
//
// Package:    SiStripMonitorCluster
// Class:      SiStripOfflineDQM
//
/**\class SiStripOfflineDQM SiStripOfflineDQM.cc
 DQM/SiStripMonitorCluster/src/SiStripOfflineDQM.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Samvel Khalatyan (ksamdev at gmail dot com)
//         Created:  Wed Oct  5 16:42:34 CET 2006
//
//

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Histograms/interface/DQMToken.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutor.h"

#include "SiStripOfflineDQM.h"

//Run Info
#include "CondFormats/RunInfo/interface/RunInfo.h"

#include <iostream>
#include <iomanip>
#include <cstdio>
#include <string>
#include <sstream>
#include <cmath>

SiStripOfflineDQM::SiStripOfflineDQM(edm::ParameterSet const& pSet)
    : actionExecutor_{pSet},
      usedWithEDMtoMEConverter_{pSet.getUntrackedParameter<bool>("UsedWithEDMtoMEConverter", false)},
      createSummary_{pSet.getUntrackedParameter<bool>("CreateSummary", false)},
      createTkMap_{pSet.getUntrackedParameter<bool>("CreateTkMap", false)},
      createTkInfoFile_{pSet.getUntrackedParameter<bool>("CreateTkInfoFile", false)},
      inputFileName_{pSet.getUntrackedParameter<std::string>("InputFileName", "")},
      outputFileName_{pSet.getUntrackedParameter<std::string>("OutputFileName", "")},
      globalStatusFilling_{pSet.getUntrackedParameter<int>("GlobalStatusFilling", 1)},
      printFaultyModuleList_{pSet.getUntrackedParameter<bool>("PrintFaultyModuleList", false)},
      detCablingToken_{globalStatusFilling_ > 0 || createTkMap_
                           ? decltype(detCablingToken_){esConsumes<edm::Transition::EndRun>()}
                           : decltype(detCablingToken_){}},
      tTopoToken_{globalStatusFilling_ > 0 || createTkMap_
                      ? decltype(tTopoToken_){esConsumes<edm::Transition::EndRun>()}
                      : decltype(tTopoToken_){}},
      tkDetMapToken_{globalStatusFilling_ > 0 || createTkMap_
                         ? decltype(tkDetMapToken_){esConsumes<edm::Transition::EndRun>()}
                         : decltype(tkDetMapToken_){}},
      geomDetToken_{createTkMap_ && createTkInfoFile_ ? decltype(geomDetToken_){esConsumes<edm::Transition::EndRun>()}
                                                      : decltype(geomDetToken_){}},
      runInfoToken_{esConsumes<edm::Transition::BeginRun>()} {
  if (createTkMap_) {
    using QualityToken = edm::ESGetToken<SiStripQuality, SiStripQualityRcd>;
    for (const auto& ps : pSet.getUntrackedParameter<std::vector<edm::ParameterSet>>("TkMapOptions")) {
      edm::ParameterSet tkMapPSet = ps;
      const auto map_type = ps.getUntrackedParameter<std::string>("mapName", "");
      tkMapPSet.augment(pSet.getUntrackedParameter<edm::ParameterSet>("TkmapParameters"));
      const bool useSSQ = tkMapPSet.getUntrackedParameter<bool>("useSSQuality", false);
      auto token = useSSQ ? QualityToken{esConsumes<edm::Transition::EndRun>(
                                edm::ESInputTag{"", tkMapPSet.getUntrackedParameter<std::string>("ssqLabel", "")})}
                          : QualityToken{};
      tkMapOptions_.emplace_back(map_type, std::move(tkMapPSet), useSSQ, std::move(token));
    }
  }

  if (createTkInfoFile_) {
    tkinfoTree_ = edm::Service<TFileService> { } -> make<TTree>("TkDetIdInfo", ""); }

  // explicit dependency to make sure the QTest reults needed here are present
  // already in endRun.
  consumes<DQMToken, edm::InRun>(edm::InputTag("siStripQTester", "DQMGenerationQTestRun"));
  consumes<DQMToken, edm::InLumi>(edm::InputTag("siStripQTester", "DQMGenerationQTestLumi"));
  usesResource("DQMStore");
  produces<DQMToken, edm::Transition::EndRun>("DQMGenerationSiStripAnalyserRun");
  produces<DQMToken, edm::Transition::EndLuminosityBlock>("DQMGenerationSiStripAnalyserLumi");
}

void SiStripOfflineDQM::beginJob() {
  // Essential: reads xml file to get the histogram names to create summary
  // Read the summary configuration file
  if (createSummary_) {
    if (!actionExecutor_.readConfiguration()) {
      edm::LogInfo("ReadConfigurationProblem") << "SiStripOfflineDQM:: Error to read configuration file!! Summary "
                                                  "will not be produced!!!";
      createSummary_ = false;
    }
  }
  edm::LogInfo("BeginJobDone") << "SiStripOfflineDQM::beginJob done";
}

void SiStripOfflineDQM::beginRun(edm::Run const& run, edm::EventSetup const& eSetup) {
  edm::LogInfo("BeginRun") << "SiStripOfflineDQM:: Begining of Run";

  int nFEDs = 0;
  if (eSetup.tryToGet<RunInfoRcd>()) {
    if (auto sumFED = eSetup.getHandle(runInfoToken_)) {
      constexpr int siStripFedIdMin{FEDNumbering::MINSiStripFEDID};
      constexpr int siStripFedIdMax{FEDNumbering::MAXSiStripFEDID};

      for (auto const fedID : sumFED->m_fed_in) {
        if (fedID >= siStripFedIdMin && fedID <= siStripFedIdMax)
          ++nFEDs;
      }
    }
  }
  auto& dqm_store = *edm::Service<DQMStore>{};
  trackerFEDsFound_ = (nFEDs > 0);
  if (!usedWithEDMtoMEConverter_) {
    if (!openInputFile(dqm_store))
      createSummary_ = false;
  }
  if (globalStatusFilling_ > 0) {
    actionExecutor_.createStatus(dqm_store);
  }
}

void SiStripOfflineDQM::produce(edm::Event&, edm::EventSetup const&) {}

void SiStripOfflineDQM::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& iSetup) {
  edm::LogInfo("EndLumiBlock") << "SiStripOfflineDQM::endLuminosityBlock";
  if (trackerFEDsFound_) {
    if (globalStatusFilling_ > 0) {
      auto& dqm_store = *edm::Service<DQMStore>{};
      actionExecutor_.fillStatusAtLumi(dqm_store);
    }
  }
}

void SiStripOfflineDQM::endRun(edm::Run const& run, edm::EventSetup const& eSetup) {
  edm::LogInfo("EndOfRun") << "SiStripOfflineDQM::endRun";

  auto& dqm_store = *edm::Service<DQMStore>{};
  if (globalStatusFilling_ > 0) {
    actionExecutor_.createStatus(dqm_store);
    if (!trackerFEDsFound_) {
      actionExecutor_.fillDummyStatus();
      return;
    }
    // Fill Global Status
    actionExecutor_.fillStatus(
        dqm_store, &eSetup.getData(detCablingToken_), &eSetup.getData(tkDetMapToken_), &eSetup.getData(tTopoToken_));
  }

  if (usedWithEDMtoMEConverter_)
    return;

  // create Summary Plots
  if (createSummary_)
    actionExecutor_.createSummaryOffline(dqm_store);

  // Create TrackerMap
  if (createTkMap_) {
    if (actionExecutor_.readTkMapConfiguration(
            &eSetup.getData(detCablingToken_), &eSetup.getData(tkDetMapToken_), &eSetup.getData(tTopoToken_))) {
      std::vector<std::string> mapNames;
      for (const auto& mapOptions : tkMapOptions_) {
        edm::LogInfo("TkMapParameters") << mapOptions.pset;
        std::string map_type = mapOptions.type;
        actionExecutor_.createOfflineTkMap(
            mapOptions.pset, dqm_store, map_type, mapOptions.useSSQ ? &eSetup.getData(mapOptions.token) : nullptr);
        mapNames.push_back(map_type);
      }
      if (createTkInfoFile_) {
        actionExecutor_.createTkInfoFile(mapNames, tkinfoTree_, dqm_store, &eSetup.getData(geomDetToken_));
      }
    }
  }
}

void SiStripOfflineDQM::endJob() {
  edm::LogInfo("EndOfJob") << "SiStripOfflineDQM::endJob";
  if (usedWithEDMtoMEConverter_)
    return;

  if (printFaultyModuleList_) {
    std::ostringstream str_val;
    auto& dqm_store = *edm::Service<DQMStore>{};
    actionExecutor_.printFaultyModuleList(dqm_store, str_val);
    std::cout << str_val.str() << std::endl;
  }
}

bool SiStripOfflineDQM::openInputFile(DQMStore& dqm_store) {
  if (inputFileName_.empty())
    return false;
  edm::LogInfo("OpenFile") << "SiStripOfflineDQM::openInputFile: Accessing root File" << inputFileName_;
  dqm_store.open(inputFileName_, false);
  return true;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripOfflineDQM);
