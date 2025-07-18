#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestRecords.h"
#include "HeterogeneousCore/AlpakaTest/interface/alpaka/AlpakaESTestData.h"

#include "TestAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * This class demonstrates a global EDProducer that
   * - consumes a device ESProduct
   * - produces a device EDProduct (that can get transferred to host automatically)
   */
  class TestAlpakaGlobalProducer : public global::EDProducer<> {
  public:
    TestAlpakaGlobalProducer(edm::ParameterSet const& config)
        : EDProducer<>(config),
          esToken_(esConsumes(config.getParameter<edm::ESInputTag>("eventSetupSource"))),
          esMultiToken_(esConsumes(config.getParameter<edm::ESInputTag>("eventSetupSourceMulti"))),
          deviceToken_{produces()},
          deviceTokenMulti2_{produces()},
          deviceTokenMulti3_{produces()},
          size_{config.getParameter<edm::ParameterSet>("size").getParameter<int32_t>(
              EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE))},
          size2_{config.getParameter<edm::ParameterSet>("size").getParameter<int32_t>(
              EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE))},
          size3_{config.getParameter<edm::ParameterSet>("size").getParameter<int32_t>(
              EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE))} {}

    void produce(edm::StreamID, device::Event& iEvent, device::EventSetup const& iSetup) const override {
      [[maybe_unused]] auto const& esData = iSetup.getData(esToken_);
      [[maybe_unused]] auto const& esMultiData = iSetup.getData(esMultiToken_);

      portabletest::TestDeviceCollection deviceProduct{size_, iEvent.queue()};
      portabletest::TestDeviceMultiCollection2 deviceProductMulti2{{{size_, size2_}}, iEvent.queue()};
      portabletest::TestDeviceMultiCollection3 deviceProductMulti3{{{size_, size2_, size3_}}, iEvent.queue()};

      // run the algorithm, potentially asynchronously
      algo_.fill(iEvent.queue(), deviceProduct);
      algo_.fillMulti2(iEvent.queue(), deviceProductMulti2);
      algo_.fillMulti3(iEvent.queue(), deviceProductMulti3);

      iEvent.emplace(deviceToken_, std::move(deviceProduct));
      iEvent.emplace(deviceTokenMulti2_, std::move(deviceProductMulti2));
      iEvent.emplace(deviceTokenMulti3_, std::move(deviceProductMulti3));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add("eventSetupSource", edm::ESInputTag{});
      desc.add("eventSetupSourceMulti", edm::ESInputTag{});

      edm::ParameterSetDescription psetSize;
      psetSize.add<int32_t>("alpaka_serial_sync");
      psetSize.add<int32_t>("alpaka_cuda_async");
      psetSize.add<int32_t>("alpaka_rocm_async");
      desc.add("size", psetSize);

      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::ESGetToken<AlpakaESTestDataADevice, AlpakaESTestRecordA> esToken_;
    const device::ESGetToken<AlpakaESTestDataACMultiDevice, AlpakaESTestRecordA> esMultiToken_;
    const device::EDPutToken<portabletest::TestDeviceCollection> deviceToken_;
    const device::EDPutToken<portabletest::TestDeviceMultiCollection2> deviceTokenMulti2_;
    const device::EDPutToken<portabletest::TestDeviceMultiCollection3> deviceTokenMulti3_;
    const int32_t size_;
    const int32_t size2_;
    const int32_t size3_;

    // implementation of the algorithm
    TestAlgo algo_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TestAlpakaGlobalProducer);
