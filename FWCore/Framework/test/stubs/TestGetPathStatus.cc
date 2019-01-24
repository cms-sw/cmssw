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
}

namespace edmtest {

  class TestGetPathStatus : public edm::global::EDAnalyzer<> {
  public:

    explicit TestGetPathStatus(edm::ParameterSet const& pset);
    virtual ~TestGetPathStatus() {}

    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;

private:

    std::vector<int> expectedStates_;
    std::vector<unsigned int> expectedIndexes_;

    edm::EDGetTokenT<edm::PathStatus> tokenPathStatus_;
    edm::EDGetTokenT<edm::EndPathStatus> tokenEndPathStatus_;
  };

  TestGetPathStatus::TestGetPathStatus(edm::ParameterSet const& pset) :
    expectedStates_(pset.getParameter<std::vector<int>>("expectedStates")),
    expectedIndexes_(pset.getParameter<std::vector<unsigned int>>("expectedIndexes")),
    tokenPathStatus_(consumes<edm::PathStatus>(pset.getParameter<edm::InputTag>("pathStatusTag"))),
    tokenEndPathStatus_(consumes<edm::EndPathStatus>(pset.getParameter<edm::InputTag>("endPathStatusTag"))){
  }

  void
  TestGetPathStatus::analyze(edm::StreamID, edm::Event const& event, edm::EventSetup const&) const {

    edm::Handle<edm::PathStatus> hPathStatus;
    event.getByToken(tokenPathStatus_, hPathStatus);
    *hPathStatus;

    unsigned int eventID = event.id().event();
    if (eventID < expectedStates_.size() && expectedStates_[eventID] != static_cast<int>(hPathStatus->state())) {
      std::cerr << "TestGetPathStatus::analyze unexpected path status state" << std::endl;
      abort();
    }
    if (eventID < expectedIndexes_.size() &&
        expectedIndexes_[eventID] != hPathStatus->index()) {
      std::cerr << "TestGetPathStatus::analyze unexpected path status index " << std::endl;
      abort();
    }

    edm::Handle<edm::EndPathStatus> hEndPathStatus;
    event.getByToken(tokenEndPathStatus_, hEndPathStatus);
    *hEndPathStatus;
  }
}
using edmtest::TestGetPathStatus;
DEFINE_FWK_MODULE(TestGetPathStatus);
