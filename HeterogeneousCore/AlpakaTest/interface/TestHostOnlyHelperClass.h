#ifndef HeterogeneousCore_AlpakaTest_interface_TestHostOnlyHelperClass_h
#define HeterogeneousCore_AlpakaTest_interface_TestHostOnlyHelperClass_h

#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestRecords.h"
#include "HeterogeneousCore/AlpakaTest/interface/ESTestData.h"

namespace cms::alpakatest {
  class TestHostOnlyHelperClass {
  public:
    TestHostOnlyHelperClass(edm::ParameterSet const& iConfig, edm::ConsumesCollector iC);

    static void fillPSetDescription(edm::ParameterSetDescription& iDesc);

    int run(edm::Event const& iEvent, edm::EventSetup const& iSetup) const;

  private:
    edm::EDGetTokenT<edmtest::IntProduct> const edToken_;
    edm::ESGetToken<cms::alpakatest::ESTestDataA, AlpakaESTestRecordA> const esToken_;
  };
}  // namespace cms::alpakatest

#endif
