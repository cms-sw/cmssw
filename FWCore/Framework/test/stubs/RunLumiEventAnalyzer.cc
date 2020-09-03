
#include "FWCore/Framework/test/stubs/RunLumiEventAnalyzer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <cassert>
#include <iostream>

namespace edmtest {

  RunLumiEventAnalyzer::RunLumiEventAnalyzer(edm::ParameterSet const& pset)
      : expectedRunLumisEvents0_(),
        expectedRunLumisEvents1_(),
        expectedRunLumisEvents_(&expectedRunLumisEvents0_),
        index_(0),
        verbose_(pset.getUntrackedParameter<bool>("verbose", false)),
        dumpTriggerResults_(pset.getUntrackedParameter<bool>("dumpTriggerResults", false)),
        expectedEndingIndex0_(pset.getUntrackedParameter<int>("expectedEndingIndex", -1)),
        expectedEndingIndex1_(pset.getUntrackedParameter<int>("expectedEndingIndex1", -1)),
        expectedEndingIndex_(expectedEndingIndex0_) {
    if (pset.existsAs<std::vector<unsigned int> >("expectedRunLumiEvents", false)) {
      std::vector<unsigned int> temp = pset.getUntrackedParameter<std::vector<unsigned int> >("expectedRunLumiEvents");
      expectedRunLumisEvents0_.assign(temp.begin(), temp.end());
    } else {
      expectedRunLumisEvents0_ = pset.getUntrackedParameter<std::vector<unsigned long long> >(
          "expectedRunLumiEvents", std::vector<unsigned long long>());
    }

    if (pset.existsAs<std::vector<unsigned int> >("expectedRunLumiEvents1", false)) {
      std::vector<unsigned int> temp = pset.getUntrackedParameter<std::vector<unsigned int> >("expectedRunLumiEvents1");
      expectedRunLumisEvents1_.assign(temp.begin(), temp.end());
    } else {
      expectedRunLumisEvents1_ = pset.getUntrackedParameter<std::vector<unsigned long long> >(
          "expectedRunLumiEvents1", std::vector<unsigned long long>());
    }
    if (dumpTriggerResults_) {
      triggerResultsToken_ = consumes(edm::InputTag("TriggerResults"));
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
