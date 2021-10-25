// This test module will look for history information in event data.

#include <iostream>
#include <vector>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edmtest {
  class TestHistoryKeeping : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
  public:
    explicit TestHistoryKeeping(edm::ParameterSet const& pset);

    void analyze(edm::Event const& e, edm::EventSetup const&) final;

    void beginRun(edm::Run const& r, edm::EventSetup const&) final;
    void endRun(edm::Run const& r, edm::EventSetup const&) final;

  private:
    std::vector<std::string> expectedProcesses_;
    int numberOfExpectedHLTProcessesInEachRun_;
  };  // class TestHistoryKeeping

  //--------------------------------------------------------------------
  //
  // Implementation details
  //--------------------------------------------------------------------

  TestHistoryKeeping::TestHistoryKeeping(edm::ParameterSet const& pset)
      : expectedProcesses_(pset.getParameter<std::vector<std::string> >("expected_processes")),
        numberOfExpectedHLTProcessesInEachRun_(
            pset.getParameter<int>("number_of_expected_HLT_processes_for_each_run")) {
    // Nothing to do.
  }

  void TestHistoryKeeping::beginRun(edm::Run const&, edm::EventSetup const&) {
    // At begin run, we're looking at, make sure we can get at the
    // parameter sets for any HLT processing.
  }

  void TestHistoryKeeping::analyze(edm::Event const& ev, edm::EventSetup const&) {
    for (std::vector<std::string>::const_iterator i = expectedProcesses_.begin(), e = expectedProcesses_.end(); i != e;
         ++i) {
      edm::ParameterSet ps;
      assert(ev.getProcessParameterSet(*i, ps));
      assert(!ps.empty());
      assert(ps.getParameter<std::string>("@process_name") == *i);
    }
  }

  void TestHistoryKeeping::endRun(edm::Run const&, edm::EventSetup const&) {
    // Nothing to do.
  }

}  // namespace edmtest

using edmtest::TestHistoryKeeping;
DEFINE_FWK_MODULE(TestHistoryKeeping);
