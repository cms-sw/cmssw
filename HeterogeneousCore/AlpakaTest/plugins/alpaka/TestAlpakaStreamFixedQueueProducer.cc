#include <cassert>
#include <optional>

#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/FixedQueueEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestRecords.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestData.h"

#include "TestAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * This class demonstrates a stream FixedQueueEDProducer that
   * - consumes a host EDProduct
   * - consumes a device ESProduct
   * - produces a device EDProduct (that gets transferred to host automatically if needed)
   * - optionally uses a product instance label
   *
   * Unlike stream::EDProducer, that uses a random device queue for every event,
   * stream::FixedQueueEDProducer always uses the same device queue for all the events
   * processed by a given framework stream.
   *
   * This module tests that the queue being used is always the same, when no device
   * products are consumed.
   */
  class TestAlpakaStreamFixedQueueProducer : public stream::FixedQueueEDProducer<> {
  public:
    TestAlpakaStreamFixedQueueProducer(edm::ParameterSet const& config)
        : FixedQueueEDProducer<>(config),
          size_{config.getParameter<edm::ParameterSet>("size").getParameter<int32_t>(
              EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE))},
          size2_{config.getParameter<edm::ParameterSet>("size").getParameter<int32_t>(
              EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE))},
          size3_{config.getParameter<edm::ParameterSet>("size").getParameter<int32_t>(
              EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE))} {
      getToken_ = consumes(config.getParameter<edm::InputTag>("source"));
      esToken_ = esConsumes(config.getParameter<edm::ESInputTag>("eventSetupSource"));
      devicePutToken_ = produces(config.getParameter<std::string>("productInstanceName"));
      devicePutTokenMulti2_ = produces(config.getParameter<std::string>("productInstanceName"));
      devicePutTokenMulti3_ = produces(config.getParameter<std::string>("productInstanceName"));
    }

    void beginStream(edm::StreamID sid, Queue queue) override { queue_ = queue; }

    void endStream(Queue queue) override { queue_.reset(); }

    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override {
      [[maybe_unused]] auto inpData = iEvent.getHandle(getToken_);
      [[maybe_unused]] auto const& esData = iSetup.getData(esToken_);

      auto deviceProduct = std::make_unique<portabletest::TestDeviceCollection>(iEvent.queue(), size_);
      auto deviceProductMulti2 = std::make_unique<portabletest::TestDeviceCollection2>(iEvent.queue(), size_, size2_);
      auto deviceProductMulti3 =
          std::make_unique<portabletest::TestDeviceCollection3>(iEvent.queue(), size_, size2_, size3_);

      // run the algorithm, potentially asynchronously
      algo_.fill(iEvent.queue(), *deviceProduct);
      algo_.fillMulti2(iEvent.queue(), *deviceProductMulti2);
      algo_.fillMulti3(iEvent.queue(), *deviceProductMulti3);

      iEvent.put(devicePutToken_, std::move(deviceProduct));
      iEvent.put(devicePutTokenMulti2_, std::move(deviceProductMulti2));
      iEvent.put(devicePutTokenMulti3_, std::move(deviceProductMulti3));

      assert(iEvent.queue() == *queue_);
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("source");
      desc.add("eventSetupSource", edm::ESInputTag{});
      desc.add<std::string>("productInstanceName", "");

      edm::ParameterSetDescription psetSize;
      psetSize.add<int32_t>("alpaka_serial_sync");
      psetSize.add<int32_t>("alpaka_cuda_async");
      psetSize.add<int32_t>("alpaka_rocm_async");
      desc.add("size", psetSize);

      descriptions.addWithDefaultLabel(desc);
    }

  private:
    edm::EDGetTokenT<edmtest::IntProduct> getToken_;
    device::ESGetToken<cms::alpakatest::AlpakaESTestDataB<Device>, AlpakaESTestRecordB> esToken_;
    device::EDPutToken<portabletest::TestDeviceCollection> devicePutToken_;
    device::EDPutToken<portabletest::TestDeviceCollection2> devicePutTokenMulti2_;
    device::EDPutToken<portabletest::TestDeviceCollection3> devicePutTokenMulti3_;
    const int32_t size_;
    const int32_t size2_;
    const int32_t size3_;

    // implementation of the algorithm
    TestAlgo algo_;

    // validation of the queue logic
    std::optional<Queue> queue_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TestAlpakaStreamFixedQueueProducer);
