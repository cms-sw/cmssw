#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestRecords.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestData.h"

#include "TestAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * This class demonstrates a stream EDProducer that
   * - consumes a host EDProduct
   * - consumes a device ESProduct
   * - produces a device EDProduct (that gets transferred to host automatically if needed)
   * - optionally uses a product instance label
   */
  class TestAlpakaStreamProducer : public stream::EDProducer<> {
  public:
    TestAlpakaStreamProducer(edm::ParameterSet const& config)
        : size_{config.getParameter<edm::ParameterSet>("size").getParameter<int32_t>(
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

    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override {
      [[maybe_unused]] auto inpData = iEvent.getHandle(getToken_);
      [[maybe_unused]] auto const& esData = iSetup.getData(esToken_);

      auto deviceProduct = std::make_unique<portabletest::TestDeviceCollection>(size_, iEvent.queue());
      auto deviceProductMulti2 = std::make_unique<portabletest::TestDeviceMultiCollection2>(
          portabletest::TestDeviceMultiCollection2::SizesArray{{size_, size2_}}, iEvent.queue());
      auto deviceProductMulti3 = std::make_unique<portabletest::TestDeviceMultiCollection3>(
          portabletest::TestDeviceMultiCollection3::SizesArray{{size_, size2_, size3_}}, iEvent.queue());

      // run the algorithm, potentially asynchronously
      algo_.fill(iEvent.queue(), *deviceProduct);
      algo_.fillMulti2(iEvent.queue(), *deviceProductMulti2);
      algo_.fillMulti3(iEvent.queue(), *deviceProductMulti3);

      iEvent.put(devicePutToken_, std::move(deviceProduct));
      iEvent.put(devicePutTokenMulti2_, std::move(deviceProductMulti2));
      iEvent.put(devicePutTokenMulti3_, std::move(deviceProductMulti3));
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
    device::EDPutToken<portabletest::TestDeviceMultiCollection2> devicePutTokenMulti2_;
    device::EDPutToken<portabletest::TestDeviceMultiCollection3> devicePutTokenMulti3_;
    const int32_t size_;
    const int32_t size2_;
    const int32_t size3_;

    // implementation of the algorithm
    TestAlgo algo_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TestAlpakaStreamProducer);
