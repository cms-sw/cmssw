#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceObject.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "TestAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TestAlpakaProducer : public global::EDProducer<> {
  public:
    TestAlpakaProducer(edm::ParameterSet const& config)
        : objectToken_{produces()},
          collectionToken_{produces()},
          deviceTokenMulti2_{produces()},
          deviceTokenMulti3_{produces()},
          size_{config.getParameter<int32_t>("size")},
          size2_{config.getParameter<int32_t>("size2")},
          size3_{config.getParameter<int32_t>("size3")} {}

    void produce(edm::StreamID sid, device::Event& event, device::EventSetup const&) const override {
      // run the algorithm, potentially asynchronously
      portabletest::TestDeviceCollection deviceCollection{size_, event.queue()};
      algo_.fill(event.queue(), deviceCollection);

      portabletest::TestDeviceObject deviceObject{event.queue()};
      algo_.fillObject(event.queue(), deviceObject, 5., 12., 13., 42);

      portabletest::TestDeviceCollection deviceProduct{size_, event.queue()};
      algo_.fill(event.queue(), deviceProduct);

      portabletest::TestDeviceMultiCollection2 deviceMultiProduct2{{{size_, size2_}}, event.queue()};
      algo_.fillMulti2(event.queue(), deviceMultiProduct2);

      portabletest::TestDeviceMultiCollection3 deviceMultiProduct3{{{size_, size2_, size3_}}, event.queue()};
      algo_.fillMulti3(event.queue(), deviceMultiProduct3);

      // put the asynchronous products into the event without waiting
      event.emplace(collectionToken_, std::move(deviceCollection));
      event.emplace(objectToken_, std::move(deviceObject));
      event.emplace(deviceTokenMulti2_, std::move(deviceMultiProduct2));
      event.emplace(deviceTokenMulti3_, std::move(deviceMultiProduct3));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<int32_t>("size");
      desc.add<int32_t>("size2");
      desc.add<int32_t>("size3");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::EDPutToken<portabletest::TestDeviceObject> objectToken_;
    const device::EDPutToken<portabletest::TestDeviceCollection> collectionToken_;
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
DEFINE_FWK_ALPAKA_MODULE(TestAlpakaProducer);
