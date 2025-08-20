#include "DataFormats/Portable/interface/PortableObject.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/MoveToDeviceCache.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "TestAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * This class demonstrates a global EDProducer that
   * - uses a MoveToDeviceCache to copy some host-side data to the devices of the backend.
   * - produces a device EDProduct (that can get transferred to host automatically)
   */
  class TestAlpakaGlobalProducerMoveToDeviceCache : public global::EDProducer<> {
  public:
    TestAlpakaGlobalProducerMoveToDeviceCache(edm::ParameterSet const& config)
        : EDProducer(config),
          getToken_(consumes(config.getParameter<edm::InputTag>("source"))),
          getTokenMulti2_(consumes(config.getParameter<edm::InputTag>("source"))),
          getTokenMulti3_(consumes(config.getParameter<edm::InputTag>("source"))),
          putToken_{produces()},
          putTokenMulti2_{produces()},
          putTokenMulti3_{produces()},
          // create host-side object that gets implicitly copied to all devices of the backend
          deviceCache_{
              PortableHostObject<TestAlgo::UpdateInfo>{cms::alpakatools::host(),
                                                       TestAlgo::UpdateInfo{config.getParameter<int32_t>("x"),
                                                                            config.getParameter<int32_t>("y"),
                                                                            config.getParameter<int32_t>("z")}}} {}

    void produce(edm::StreamID, device::Event& iEvent, device::EventSetup const& iSetup) const override {
      auto const& input = iEvent.get(getToken_);
      auto const& inputMulti2 = iEvent.get(getTokenMulti2_);
      auto const& inputMulti3 = iEvent.get(getTokenMulti3_);

      // get the object corresponding to the Device the Event is being processed on
      auto const& infoObj = deviceCache_.get(iEvent.queue());

      // run the algorithm, potentially asynchronously
      auto deviceProduct = algo_.update(iEvent.queue(), input, infoObj.data());
      auto deviceProductMulti2 = algo_.updateMulti2(iEvent.queue(), inputMulti2, infoObj.data());
      auto deviceProductMulti3 = algo_.updateMulti3(iEvent.queue(), inputMulti3, infoObj.data());

      iEvent.emplace(putToken_, std::move(deviceProduct));
      iEvent.emplace(putTokenMulti2_, std::move(deviceProductMulti2));
      iEvent.emplace(putTokenMulti3_, std::move(deviceProductMulti3));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;

      desc.add("source", edm::InputTag{});
      desc.add<int32_t>("x", 0);
      desc.add<int32_t>("y", 1);
      desc.add<int32_t>("z", 2);

      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::EDGetToken<portabletest::TestDeviceCollection> getToken_;
    const device::EDGetToken<portabletest::TestDeviceMultiCollection2> getTokenMulti2_;
    const device::EDGetToken<portabletest::TestDeviceMultiCollection3> getTokenMulti3_;
    const device::EDPutToken<portabletest::TestDeviceCollection> putToken_;
    const device::EDPutToken<portabletest::TestDeviceMultiCollection2> putTokenMulti2_;
    const device::EDPutToken<portabletest::TestDeviceMultiCollection3> putTokenMulti3_;

    // implementation of the algorithm
    TestAlgo algo_;

    cms::alpakatools::MoveToDeviceCache<Device, PortableHostObject<TestAlgo::UpdateInfo>> deviceCache_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TestAlpakaGlobalProducerMoveToDeviceCache);
