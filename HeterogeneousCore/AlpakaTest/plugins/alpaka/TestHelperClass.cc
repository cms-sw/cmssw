#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "TestHelperClass.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  TestHelperClass::TestHelperClass(edm::ParameterSet const& iConfig, edm::ConsumesCollector iC)
      : getToken_(iC.consumes(iConfig.getParameter<edm::InputTag>("source"))),
        esTokenHost_(iC.esConsumes()),
        esTokenDevice_(iC.esConsumes()) {}

  void TestHelperClass::fillPSetDescription(edm::ParameterSetDescription& iDesc) { iDesc.add<edm::InputTag>("source"); }

  void TestHelperClass::makeAsync(device::Event const& iEvent, device::EventSetup const& iSetup) {
    [[maybe_unused]] auto esDataHostHandle = iSetup.getHandle(esTokenHost_);
    [[maybe_unused]] auto const& esDataDevice = iSetup.getData(esTokenDevice_);
    portabletest::TestDeviceCollection const& deviceProduct = iEvent.get(getToken_);

    hostProduct_ = portabletest::TestHostCollection{deviceProduct->metadata().size(), iEvent.queue()};

    alpaka::memcpy(iEvent.queue(), hostProduct_.buffer(), deviceProduct.const_buffer());
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
