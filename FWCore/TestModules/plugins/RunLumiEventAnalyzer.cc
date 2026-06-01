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

#include <cassert>
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
    std::vector<unsigned long long> const expectedRunLumisEvents0_;
    std::vector<unsigned long long> const expectedRunLumisEvents1_;
    std::vector<unsigned long long> const* const expectedRunLumisEvents_;
    bool const verbose_;
    bool const dumpTriggerResults_;
    int const expectedEndingIndex0_;
    int const expectedEndingIndex1_;
    int const expectedEndingIndex_;
    edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
    int index_ = 0;
  };

  RunLumiEventAnalyzer::RunLumiEventAnalyzer(edm::ParameterSet const& pset)
      : expectedRunLumisEvents0_(pset.getUntrackedParameter<std::vector<unsigned long long>>("expectedRunLumiEvents")),
        expectedRunLumisEvents1_(pset.getUntrackedParameter<std::vector<unsigned long long>>("expectedRunLumiEvents1")),
        expectedRunLumisEvents_(&expectedRunLumisEvents0_),
        verbose_(pset.getUntrackedParameter<bool>("verbose")),
        dumpTriggerResults_(pset.getUntrackedParameter<bool>("dumpTriggerResults")),
        expectedEndingIndex0_(pset.getUntrackedParameter<int>("expectedEndingIndex")),
        expectedEndingIndex1_(pset.getUntrackedParameter<int>("expectedEndingIndex1")),
        expectedEndingIndex_(expectedEndingIndex0_) {
    if (dumpTriggerResults_) {
      triggerResultsToken_ = consumes(edm::InputTag("TriggerResults"));
    }
  }

  void RunLumiEventAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.addUntracked<bool>("verbose", false);
    desc.addUntracked<bool>("dumpTriggerResults", false);
    desc.addUntracked<int>("expectedEndingIndex", -1);
    desc.addUntracked<int>("expectedEndingIndex1", -1);
    desc.addUntracked<std::vector<unsigned long long>>("expectedRunLumiEvents", {});
    desc.addUntracked<std::vector<unsigned long long>>("expectedRunLumiEvents1", {});

    descriptions.addDefault(desc);
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

    if ((index_ + 2U) < expectedRunLumisEvents_->size()) {
      assert(expectedRunLumisEvents_->at(index_) == event.run());
      ++index_;
      assert(expectedRunLumisEvents_->at(index_) == event.luminosityBlock());
      ++index_;
      assert(expectedRunLumisEvents_->at(index_) == event.id().event());
      ++index_;
    }
  }

  void RunLumiEventAnalyzer::beginRun(edm::Run const& run, edm::EventSetup const&) {
    if (verbose_) {
      edm::LogAbsolute("RunLumiEvent") << "RUN_LUMI_EVENT " << run.run() << ", " << 0 << ", " << 0;
    }

    if ((index_ + 2U) < expectedRunLumisEvents_->size()) {
      assert(expectedRunLumisEvents_->at(index_) == run.run());
      ++index_;
      assert(expectedRunLumisEvents_->at(index_) == 0);
      ++index_;
      assert(expectedRunLumisEvents_->at(index_) == 0);
      ++index_;
    }
  }

  void RunLumiEventAnalyzer::endRun(edm::Run const& run, edm::EventSetup const&) {
    if (verbose_) {
      edm::LogAbsolute("RunLumiEvent") << "RUN_LUMI_EVENT " << run.run() << ", " << 0 << ", " << 0;
    }

    if ((index_ + 2U) < expectedRunLumisEvents_->size()) {
      if (!(expectedRunLumisEvents_->at(index_) == run.run())) {
        throw cms::Exception("UnexpectedRun") << "RunLumiEventAnalyzer::endRun unexpected run\n"
                                                 "  expected "
                                              << expectedRunLumisEvents_->at(index_) << "  found    " << run.run();
      }
      ++index_;
      if (!(expectedRunLumisEvents_->at(index_) == 0)) {
        throw cms::Exception("UnexpectedLumi") << "RunLumiEventAnalyzer::endRun unexpected lumi\n"
                                                  "  expected "
                                               << expectedRunLumisEvents_->at(index_) << "  found    0";
      }
      ++index_;
      if (!(expectedRunLumisEvents_->at(index_) == 0)) {
        throw cms::Exception("UnexpectedEvent") << "RunLumiEventAnalyzer::endRun unexpected event\n"
                                                   "  expected "
                                                << expectedRunLumisEvents_->at(index_) << "  found    0";
      }
      ++index_;
    }
  }

  void RunLumiEventAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const&) {
    if (verbose_) {
      edm::LogAbsolute("RunLumiEvent") << "RUN_LUMI_EVENT " << lumi.run() << ", " << lumi.luminosityBlock() << ", "
                                       << 0;
    }

    if ((index_ + 2U) < expectedRunLumisEvents_->size()) {
      assert(expectedRunLumisEvents_->at(index_) == lumi.run());
      ++index_;
      assert(expectedRunLumisEvents_->at(index_) == lumi.luminosityBlock());
      ++index_;
      assert(expectedRunLumisEvents_->at(index_) == 0);
      ++index_;
    }
  }

  void RunLumiEventAnalyzer::endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const&) {
    if (verbose_) {
      edm::LogAbsolute("RunLumiEvent") << "RUN_LUMI_EVENT " << lumi.run() << ", " << lumi.luminosityBlock() << ", "
                                       << 0;
    }

    if ((index_ + 2U) < expectedRunLumisEvents_->size()) {
      if (!(expectedRunLumisEvents_->at(index_) == lumi.run())) {
        throw cms::Exception("UnexpectedRun") << "RunLumiEventAnalyzer::endLuminosityBlock unexpected run\n"
                                                 "  expected "
                                              << expectedRunLumisEvents_->at(index_) << "  found    " << lumi.run();
      }
      ++index_;
      if (!(expectedRunLumisEvents_->at(index_) == lumi.luminosityBlock())) {
        throw cms::Exception("UnexpectedLumi")
            << "RunLumiEventAnalyzer::endLuminosityBlock unexpected lumi"
               "  expected "
            << expectedRunLumisEvents_->at(index_) << "  found    " << lumi.luminosityBlock();
      }
      ++index_;
      if (!(expectedRunLumisEvents_->at(index_) == 0)) {
        throw cms::Exception("UnexpectedEvent") << "RunLumiEventAnalyzer::endLuminosityBlock unexpected event"
                                                   "  expected "
                                                << expectedRunLumisEvents_->at(index_) << "  found    0";
      }
      ++index_;
    }
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
