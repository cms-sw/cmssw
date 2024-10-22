
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutor.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQM/SiStripMonitorSummary/interface/SiStripClassToMonitorCondData.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>
#include <sstream>
#include <string>

class SiStripWebInterface;
class SiStripFedCabling;
class SiStripDetCabling;
class SiStripClassToMonitorCondData;
class FEDRawDataCollection;

class SiStripAnalyser
    : public edm::one::EDAnalyzer<edm::one::SharedResources, edm::one::WatchRuns, edm::one::WatchLuminosityBlocks> {
public:
  typedef dqm::harvesting::MonitorElement MonitorElement;
  typedef dqm::harvesting::DQMStore DQMStore;

  SiStripAnalyser(const edm::ParameterSet& ps);
  ~SiStripAnalyser() override;

private:
  void beginJob() override;
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup) override;
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup) override;
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) override;
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) override;
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup) override;
  void endJob() override;

  void checkTrackerFEDs(edm::Event const& e);

  SiStripClassToMonitorCondData condDataMon_;
  SiStripActionExecutor actionExecutor_;
  edm::ParameterSet tkMapPSet_;

  int summaryFrequency_{-1};
  int staticUpdateFrequency_;
  int globalStatusFilling_;
  int shiftReportFrequency_;

  edm::EDGetTokenT<FEDRawDataCollection> rawDataToken_;

  std::string outputFilePath_;
  std::string outputFileName_;

  const SiStripDetCabling* detCabling_;
  edm::ESGetToken<SiStripDetCabling, SiStripDetCablingRcd> detCablingToken_;
  edm::ESWatcher<SiStripFedCablingRcd> fedCablingWatcher_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_, tTopoTokenELB_, tTopoTokenBR_;
  edm::ESGetToken<TkDetMap, TrackerTopologyRcd> tkDetMapToken_, tkDetMapTokenELB_, tkDetMapTokenBR_;
  edm::ESGetToken<SiStripQuality, SiStripQualityRcd> stripQualityToken_;

  int nLumiSecs_{};
  int nEvents_{};
  bool trackerFEDsFound_{false};
  bool printFaultyModuleList_;
  bool endLumiAnalysisOn_{false};
};

SiStripAnalyser::SiStripAnalyser(edm::ParameterSet const& ps)
    : condDataMon_{ps, consumesCollector()},
      actionExecutor_{ps},
      tkMapPSet_{ps.getParameter<edm::ParameterSet>("TkmapParameters")},
      summaryFrequency_{ps.getUntrackedParameter<int>("SummaryCreationFrequency", 1)},
      staticUpdateFrequency_{ps.getUntrackedParameter<int>("StaticUpdateFrequency", 1)},
      globalStatusFilling_{ps.getUntrackedParameter<int>("GlobalStatusFilling", 1)},
      shiftReportFrequency_{ps.getUntrackedParameter<int>("ShiftReportFrequency", 1)},
      rawDataToken_{consumes<FEDRawDataCollection>(ps.getUntrackedParameter<edm::InputTag>("RawDataTag"))},
      detCablingToken_(esConsumes<edm::Transition::BeginRun>()),
      tTopoToken_(esConsumes()),
      tTopoTokenELB_(esConsumes<edm::Transition::EndLuminosityBlock>()),
      tTopoTokenBR_(esConsumes<edm::Transition::BeginRun>()),
      tkDetMapToken_(esConsumes()),
      tkDetMapTokenELB_(esConsumes<edm::Transition::EndLuminosityBlock>()),
      tkDetMapTokenBR_(esConsumes<edm::Transition::BeginRun>()),
      printFaultyModuleList_{ps.getUntrackedParameter<bool>("PrintFaultyModuleList", true)} {
  usesResource("DQMStore");
  std::string const localPath{"DQM/SiStripMonitorClient/test/loader.html"};
  std::ifstream fin{edm::FileInPath(localPath).fullPath(), std::ios::in};
  if (!fin) {
    std::cerr << "Input File: loader.html"
              << " could not be opened!" << std::endl;
    return;
  }
  edm::LogInfo("SiStripAnalyser") << " SiStripAnalyser::Creating SiStripAnalyser ";
}

SiStripAnalyser::~SiStripAnalyser() { edm::LogInfo("SiStripAnalyser") << "SiStripAnalyser::Deleting SiStripAnalyser "; }

void SiStripAnalyser::beginJob() {
  // Read the summary configuration file
  if (!actionExecutor_.readConfiguration()) {
    edm::LogInfo("SiStripAnalyser") << "SiStripAnalyser:: Error to read configuration file!! Summary will "
                                       "not be produced!!!";
  }
}

void SiStripAnalyser::beginRun(edm::Run const& run, edm::EventSetup const& eSetup) {
  edm::LogInfo("SiStripAnalyser") << "SiStripAnalyser:: Begining of Run";

  // Check latest Fed cabling and create TrackerMapCreator
  if (fedCablingWatcher_.check(eSetup)) {
    edm::LogInfo("SiStripAnalyser") << "SiStripAnalyser::beginRun: "
                                    << " Change in Cabling, recrated TrackerMap";
    detCabling_ = &eSetup.getData(detCablingToken_);
    if (!actionExecutor_.readTkMapConfiguration(
            detCabling_, &eSetup.getData(tkDetMapTokenBR_), &eSetup.getData(tTopoTokenBR_))) {
      edm::LogInfo("SiStripAnalyser") << "SiStripAnalyser:: Error to read configuration file!! TrackerMap "
                                         "will not be produced!!!";
    }
  }
  condDataMon_.beginRun(run.run(), eSetup);
  if (globalStatusFilling_) {
    auto& dqm_store = *edm::Service<DQMStore>{};
    actionExecutor_.createStatus(dqm_store);
  }
}

void SiStripAnalyser::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) {
  edm::LogInfo("SiStripAnalyser") << "SiStripAnalyser:: Begin of LS transition";
}

void SiStripAnalyser::analyze(edm::Event const& e, edm::EventSetup const& eSetup) {
  ++nEvents_;
  if (nEvents_ == 1 && globalStatusFilling_ > 0) {
    checkTrackerFEDs(e);
    if (!trackerFEDsFound_) {
      actionExecutor_.fillDummyStatus();
      actionExecutor_.createDummyShiftReport();
    } else {
      auto& dqm_store = *edm::Service<DQMStore>{};
      actionExecutor_.fillStatus(dqm_store, detCabling_, &eSetup.getData(tkDetMapToken_), &eSetup.getData(tTopoToken_));
      if (shiftReportFrequency_ != -1)
        actionExecutor_.createShiftReport(dqm_store);
    }
  }
}

void SiStripAnalyser::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) {
  edm::LogInfo("SiStripAnalyser") << "SiStripAnalyser:: End of LS transition, "
                                     "performing the DQM client operation";
  ++nLumiSecs_;

  if (!trackerFEDsFound_) {
    actionExecutor_.fillDummyStatus();
    return;
  }
  endLumiAnalysisOn_ = true;

  std::cout << "====================================================== " << std::endl;
  std::cout << " ===> Iteration # " << nLumiSecs_ << " " << lumiSeg.luminosityBlock() << std::endl;
  std::cout << "====================================================== " << std::endl;

  auto& dqm_store = *edm::Service<DQMStore>{};
  // Fill Global Status
  if (globalStatusFilling_ > 0) {
    actionExecutor_.fillStatus(
        dqm_store, detCabling_, &eSetup.getData(tkDetMapTokenELB_), &eSetup.getData(tTopoTokenELB_));
  }
  // -- Create summary monitor elements according to the frequency
  if (summaryFrequency_ != -1 && nLumiSecs_ > 0 && nLumiSecs_ % summaryFrequency_ == 0) {
    std::cout << " Creating Summary " << std::endl;
    actionExecutor_.createSummary(dqm_store);
  }
  endLumiAnalysisOn_ = false;
}

void SiStripAnalyser::endRun(edm::Run const& run, edm::EventSetup const& eSetup) {
  edm::LogInfo("SiStripAnalyser") << "SiStripAnalyser:: End of Run";
}

void SiStripAnalyser::endJob() {
  edm::LogInfo("SiStripAnalyser") << "SiStripAnalyser:: endjob called!";
  if (printFaultyModuleList_) {
    std::ostringstream str_val;
    auto& dqm_store = *edm::Service<DQMStore>{};
    actionExecutor_.printFaultyModuleList(dqm_store, str_val);
    std::cout << str_val.str() << std::endl;
  }
}

void SiStripAnalyser::checkTrackerFEDs(edm::Event const& e) {
  edm::Handle<FEDRawDataCollection> rawDataHandle;
  e.getByToken(rawDataToken_, rawDataHandle);
  if (!rawDataHandle.isValid())
    return;

  auto const& rawDataCollection = *rawDataHandle;
  constexpr int siStripFedIdMin{FEDNumbering::MINSiStripFEDID};
  constexpr int siStripFedIdMax{FEDNumbering::MAXSiStripFEDID};

  for (int i = siStripFedIdMin; i <= siStripFedIdMax; ++i) {
    auto const& fedData = rawDataCollection.FEDData(i);
    if (fedData.size() && fedData.data()) {
      trackerFEDsFound_ = true;
      return;
    }
  }
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripAnalyser);
