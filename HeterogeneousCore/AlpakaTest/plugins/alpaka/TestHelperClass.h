#ifndef HeterogeneousCore_AlpakaTest_plugins_alpaka_TestHelperClass_h
#define HeterogeneousCore_AlpakaTest_plugins_alpaka_TestHelperClass_h

#include "DataFormats/PortableTestObjects/interface/TestHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestRecords.h"
#include "HeterogeneousCore/AlpakaTest/interface/ESTestData.h"
#include "HeterogeneousCore/AlpakaTest/interface/alpaka/AlpakaESTestData.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class TestHelperClass {
  public:
    TestHelperClass(edm::ParameterSet const& iConfig, edm::ConsumesCollector iC);

    static void fillPSetDescription(edm::ParameterSetDescription& iDesc);

    void makeAsync(device::Event const& iEvent, device::EventSetup const& iSetup);

    portabletest::TestHostCollection moveFrom() { return std::move(hostProduct_); }

  private:
    const device::EDGetToken<portabletest::TestDeviceCollection> getToken_;
    const edm::ESGetToken<cms::alpakatest::ESTestDataA, AlpakaESTestRecordA> esTokenHost_;
    const device::ESGetToken<AlpakaESTestDataCDevice, AlpakaESTestRecordC> esTokenDevice_;

    // hold the output product between acquire() and produce()
    portabletest::TestHostCollection hostProduct_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
