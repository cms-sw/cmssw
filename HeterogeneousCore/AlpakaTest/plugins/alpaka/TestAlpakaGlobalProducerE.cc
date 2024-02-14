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
   * - consumes a device EDProduct
   * - produces a device EDProduct (that can get transferred to host automatically)
   */
  class TestAlpakaGlobalProducerE : public global::EDProducer<> {
  public:
    TestAlpakaGlobalProducerE(edm::ParameterSet const& config)
        : esToken_(esConsumes(config.getParameter<edm::ESInputTag>("eventSetupSource"))),
          getToken_(consumes(config.getParameter<edm::InputTag>("source"))),
          getTokenMulti2_(consumes(config.getParameter<edm::InputTag>("source"))),
          getTokenMulti3_(consumes(config.getParameter<edm::InputTag>("source"))),
          putToken_{produces()},
          putTokenMulti2_{produces()},
          putTokenMulti3_{produces()} {}

    void produce(edm::StreamID, device::Event& iEvent, device::EventSetup const& iSetup) const override {
      auto const& esData = iSetup.getData(esToken_);
      auto const& input = iEvent.get(getToken_);
      auto const& inputMulti2 = iEvent.get(getTokenMulti2_);
      auto const& inputMulti3 = iEvent.get(getTokenMulti3_);

      // run the algorithm, potentially asynchronously
      auto deviceProduct = algo_.update(iEvent.queue(), input, esData);
      auto deviceProductMulti2 = algo_.updateMulti2(iEvent.queue(), inputMulti2, esData);
      auto deviceProductMulti3 = algo_.updateMulti3(iEvent.queue(), inputMulti3, esData);

      iEvent.emplace(putToken_, std::move(deviceProduct));
      iEvent.emplace(putTokenMulti2_, std::move(deviceProductMulti2));
      iEvent.emplace(putTokenMulti3_, std::move(deviceProductMulti3));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add("eventSetupSource", edm::ESInputTag{});
      desc.add("source", edm::InputTag{});

      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::ESGetToken<AlpakaESTestDataEDevice, AlpakaESTestRecordC> esToken_;
    const device::EDGetToken<portabletest::TestDeviceCollection> getToken_;
    const device::EDGetToken<portabletest::TestDeviceMultiCollection2> getTokenMulti2_;
    const device::EDGetToken<portabletest::TestDeviceMultiCollection3> getTokenMulti3_;
    const device::EDPutToken<portabletest::TestDeviceCollection> putToken_;
    const device::EDPutToken<portabletest::TestDeviceMultiCollection2> putTokenMulti2_;
    const device::EDPutToken<portabletest::TestDeviceMultiCollection3> putTokenMulti3_;

    // implementation of the algorithm
    TestAlgo algo_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TestAlpakaGlobalProducerE);
