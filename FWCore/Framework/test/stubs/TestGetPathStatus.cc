#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DataFormats/Common/interface/EndPathStatus.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/PathStatus.h"

#include <iostream>

namespace edm {
  class EventSetup;
  class StreamID;
}  // namespace edm

namespace edmtest {

  class TestGetPathStatus : public edm::global::EDAnalyzer<> {
  public:
    explicit TestGetPathStatus(edm::ParameterSet const& pset);
    ~TestGetPathStatus() override {}

    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;

  private:
    std::vector<int> expectedStates_;
    std::vector<unsigned int> expectedIndexes_;

    edm::EDGetTokenT<edm::PathStatus> tokenPathStatus_;
    edm::EDGetTokenT<edm::EndPathStatus> tokenEndPathStatus_;
  };

  TestGetPathStatus::TestGetPathStatus(edm::ParameterSet const& pset)
      : expectedStates_(pset.getParameter<std::vector<int>>("expectedStates")),
        expectedIndexes_(pset.getParameter<std::vector<unsigned int>>("expectedIndexes")),
        tokenPathStatus_(consumes<edm::PathStatus>(pset.getParameter<edm::InputTag>("pathStatusTag"))),
        tokenEndPathStatus_(consumes<edm::EndPathStatus>(pset.getParameter<edm::InputTag>("endPathStatusTag"))) {}

  void TestGetPathStatus::analyze(edm::StreamID, edm::Event const& event, edm::EventSetup const&) const {
    auto const& pathStatus = event.get(tokenPathStatus_);

    unsigned int eventID = event.id().event();
    if (eventID < expectedStates_.size() && expectedStates_[eventID] != static_cast<int>(pathStatus.state())) {
      std::cerr << "TestGetPathStatus::analyze unexpected path status state" << std::endl;
      abort();
    }
    if (eventID < expectedIndexes_.size() && expectedIndexes_[eventID] != pathStatus.index()) {
      std::cerr << "TestGetPathStatus::analyze unexpected path status index " << std::endl;
      abort();
    }

    (void)event.get(tokenEndPathStatus_);
  }
}  // namespace edmtest
using edmtest::TestGetPathStatus;
DEFINE_FWK_MODULE(TestGetPathStatus);
