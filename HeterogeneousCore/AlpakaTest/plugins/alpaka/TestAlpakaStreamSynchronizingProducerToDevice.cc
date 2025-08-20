#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "TestAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * This class demonstrates a stream EDProducer that
   * - produces a device EDProduct (that can get transferred to host automatically)
   * - synchronizes in a non-blocking way with the ExternalWork module
   *   ability (via the SynchronizingEDProcucer base class)
   */
  class TestAlpakaStreamSynchronizingProducerToDevice : public stream::SynchronizingEDProducer<> {
  public:
    TestAlpakaStreamSynchronizingProducerToDevice(edm::ParameterSet const& iConfig)
        : SynchronizingEDProducer<>(iConfig),
          putToken_{produces()},
          size_{iConfig.getParameter<edm::ParameterSet>("size").getParameter<int32_t>(
              EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE))} {}

    void acquire(device::Event const& iEvent, device::EventSetup const& iSetup) override {
      deviceProduct_ = std::make_unique<portabletest::TestDeviceCollection>(size_, iEvent.queue());

      // run the algorithm, potentially asynchronously
      algo_.fill(iEvent.queue(), *deviceProduct_);
    }

    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override {
      iEvent.put(putToken_, std::move(deviceProduct_));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;

      edm::ParameterSetDescription psetSize;
      psetSize.add<int32_t>("alpaka_serial_sync");
      psetSize.add<int32_t>("alpaka_cuda_async");
      psetSize.add<int32_t>("alpaka_rocm_async");
      desc.add("size", psetSize);

      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::EDPutToken<portabletest::TestDeviceCollection> putToken_;
    const int32_t size_;

    // implementation of the algorithm
    TestAlgo algo_;

    std::unique_ptr<portabletest::TestDeviceCollection> deviceProduct_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TestAlpakaStreamSynchronizingProducerToDevice);
