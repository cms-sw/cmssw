#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "TestHelperClass.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  TestHelperClass::TestHelperClass(edm::ParameterSet const& iConfig, edm::ConsumesCollector iC)
      : getToken_(iC.consumes(iConfig.getParameter<edm::InputTag>("source"))),
        getTokenMulti2_(iC.consumes(iConfig.getParameter<edm::InputTag>("source"))),
        getTokenMulti3_(iC.consumes(iConfig.getParameter<edm::InputTag>("source"))),
        esTokenHost_(iC.esConsumes()),
        esTokenDevice_(iC.esConsumes()) {}

  void TestHelperClass::fillPSetDescription(edm::ParameterSetDescription& iDesc) { iDesc.add<edm::InputTag>("source"); }

  void TestHelperClass::makeAsync(device::Event const& iEvent, device::EventSetup const& iSetup) {
    [[maybe_unused]] auto esDataHostHandle = iSetup.getHandle(esTokenHost_);
    [[maybe_unused]] auto const& esDataDevice = iSetup.getData(esTokenDevice_);
    portabletest::TestDeviceCollection const& deviceProduct = iEvent.get(getToken_);
    portabletest::TestDeviceMultiCollection2 const& deviceProductMulti2 = iEvent.get(getTokenMulti2_);
    portabletest::TestDeviceMultiCollection3 const& deviceProductMulti3 = iEvent.get(getTokenMulti3_);

    hostProduct_ = portabletest::TestHostCollection{deviceProduct->metadata().size(), iEvent.queue()};
    hostProductMulti2_ = portabletest::TestHostMultiCollection2{deviceProductMulti2.sizes(), iEvent.queue()};
    hostProductMulti3_ = portabletest::TestHostMultiCollection3{deviceProductMulti3.sizes(), iEvent.queue()};

    alpaka::memcpy(iEvent.queue(), hostProduct_->buffer(), deviceProduct.const_buffer());
    alpaka::memcpy(iEvent.queue(), hostProductMulti2_->buffer(), deviceProductMulti2.const_buffer());
    alpaka::memcpy(iEvent.queue(), hostProductMulti3_->buffer(), deviceProductMulti3.const_buffer());
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
