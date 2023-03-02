#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaTest/interface/TestHostOnlyHelperClass.h"

namespace cms::alpakatest {
  TestHostOnlyHelperClass::TestHostOnlyHelperClass(edm::ParameterSet const& iConfig, edm::ConsumesCollector iC)
      : edToken_(iC.consumes(iConfig.getParameter<edm::InputTag>("intSource"))), esToken_(iC.esConsumes()) {}

  void TestHostOnlyHelperClass::fillPSetDescription(edm::ParameterSetDescription& iDesc) {
    iDesc.add<edm::InputTag>("intSource");
  }

  int TestHostOnlyHelperClass::run(edm::Event const& iEvent, edm::EventSetup const& iSetup) const {
    auto const& ed = iEvent.get(edToken_);
    auto const& es = iSetup.getData(esToken_);

    return ed.value + es.value();
  }
}  // namespace cms::alpakatest
