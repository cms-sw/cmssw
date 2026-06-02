#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <string_view>
#include <vector>

namespace edmtest {
  class RunLumiEventAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::WatchLuminosityBlocks> {
  public:
    explicit RunLumiEventAnalyzer(edm::ParameterSet const& pset);

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    void analyze(edm::Event const& event, edm::EventSetup const& es) final;
    void beginRun(edm::Run const& run, edm::EventSetup const& es) final;
    void endRun(edm::Run const& run, edm::EventSetup const& es) final;
    void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& es) final;
    void endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& es) final;
    void endJob();

  private:
    void checkRunLumiEvent(std::string_view transition,
                           edm::RunNumber_t run,
                           edm::LuminosityBlockNumber_t lumi,
                           edm::EventNumber_t event);

    std::vector<unsigned long long> const expectedRunLumisEvents_;
    bool const verbose_;
    bool const dumpTriggerResults_;
    int const expectedEndingIndex_;
    edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
    int index_ = 0;
  };

  RunLumiEventAnalyzer::RunLumiEventAnalyzer(edm::ParameterSet const& pset)
      : expectedRunLumisEvents_(pset.getUntrackedParameter<std::vector<unsigned long long>>("expectedRunLumiEvents")),
        verbose_(pset.getUntrackedParameter<bool>("verbose")),
        dumpTriggerResults_(pset.getUntrackedParameter<bool>("dumpTriggerResults")),
        expectedEndingIndex_(pset.getUntrackedParameter<int>("expectedEndingIndex")) {
    if (dumpTriggerResults_) {
      triggerResultsToken_ = consumes(edm::InputTag("TriggerResults"));
    }
  }

  void RunLumiEventAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.addUntracked<bool>("verbose", false);
    desc.addUntracked<bool>("dumpTriggerResults", false);
    desc.addUntracked<int>("expectedEndingIndex", -1);
    desc.addUntracked<std::vector<unsigned long long>>("expectedRunLumiEvents", {});

    descriptions.addDefault(desc);
  }

  void RunLumiEventAnalyzer::checkRunLumiEvent(std::string_view transition,
                                               edm::RunNumber_t run,
                                               edm::LuminosityBlockNumber_t lumi,
                                               edm::EventNumber_t event) {
    if ((index_ + 2U) < expectedRunLumisEvents_.size()) {
      if (expectedRunLumisEvents_.at(index_) != run) {
        throw cms::Exception("Assert").format("{} unexpected Run: expected {} found {} (index {})",
                                              transition,
                                              expectedRunLumisEvents_.at(index_),
                                              run,
                                              index_);
      }
      ++index_;
      if (expectedRunLumisEvents_.at(index_) != lumi) {
        throw cms::Exception("Assert").format("{} unexpected LuminosityBlock: expected {} found {} (index {})",
                                              transition,
                                              expectedRunLumisEvents_.at(index_),
                                              lumi,
                                              index_);
      }
      ++index_;
      if (expectedRunLumisEvents_.at(index_) != event) {
        throw cms::Exception("Assert").format("{} unexpected Event: expected {} found {} (index {})",
                                              transition,
                                              expectedRunLumisEvents_.at(index_),
                                              event,
                                              index_);
      }
      ++index_;
    }
  }

  void RunLumiEventAnalyzer::analyze(edm::Event const& event, edm::EventSetup const&) {
    if (verbose_) {
      edm::LogAbsolute("RunLumiEvent") << "RUN_LUMI_EVENT " << event.run() << ", " << event.luminosityBlock() << ", "
                                       << event.id().event();
    }

    if (dumpTriggerResults_) {
      if (auto triggerResults = event.getHandle(triggerResultsToken_)) {
        edm::LogAbsolute("RunLumiEvent") << "TestFailuresAnalyzer dumping TriggerResults";
        edm::LogAbsolute("RunLumiEvent") << *triggerResults;
      } else {
        edm::LogAbsolute("RunLumiEvent") << "TriggerResults not found\n";
      }
    }

    checkRunLumiEvent("analyze", event.run(), event.luminosityBlock(), event.id().event());
  }

  void RunLumiEventAnalyzer::beginRun(edm::Run const& run, edm::EventSetup const&) {
    if (verbose_) {
      edm::LogAbsolute("RunLumiEvent") << "RUN_LUMI_EVENT " << run.run() << ", " << 0 << ", " << 0;
    }

    checkRunLumiEvent("beginRun", run.run(), 0, 0);
  }

  void RunLumiEventAnalyzer::endRun(edm::Run const& run, edm::EventSetup const&) {
    if (verbose_) {
      edm::LogAbsolute("RunLumiEvent") << "RUN_LUMI_EVENT " << run.run() << ", " << 0 << ", " << 0;
    }

    checkRunLumiEvent("endRun", run.run(), 0, 0);
  }

  void RunLumiEventAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const&) {
    if (verbose_) {
      edm::LogAbsolute("RunLumiEvent") << "RUN_LUMI_EVENT " << lumi.run() << ", " << lumi.luminosityBlock() << ", "
                                       << 0;
    }

    checkRunLumiEvent("beginLuminosityBlock", lumi.run(), lumi.luminosityBlock(), 0);
  }

  void RunLumiEventAnalyzer::endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const&) {
    if (verbose_) {
      edm::LogAbsolute("RunLumiEvent") << "RUN_LUMI_EVENT " << lumi.run() << ", " << lumi.luminosityBlock() << ", "
                                       << 0;
    }

    checkRunLumiEvent("endLuminosityBlock", lumi.run(), lumi.luminosityBlock(), 0);
  }

  void RunLumiEventAnalyzer::endJob() {
    if (expectedEndingIndex_ != -1 && index_ != expectedEndingIndex_) {
      throw cms::Exception("UnexpectedEvent",
                           "RunLumiEventAnalyzer::endJob. Unexpected number of runs, lumis, and events");
    }
  }

}  // namespace edmtest
using edmtest::RunLumiEventAnalyzer;
DEFINE_FWK_MODULE(RunLumiEventAnalyzer);
