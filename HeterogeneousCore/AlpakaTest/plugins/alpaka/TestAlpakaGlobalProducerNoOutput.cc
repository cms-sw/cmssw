#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * This EDProducer only consumes a device EDProduct, and is intended
   * only for testing purposes. Do not use it as an example.
   */
  class TestAlpakaGlobalProducerNoOutput : public global::EDProducer<> {
  public:
    TestAlpakaGlobalProducerNoOutput(edm::ParameterSet const& config)
        : EDProducer<>(config), getToken_(consumes(config.getParameter<edm::InputTag>("source"))) {}

    void produce(edm::StreamID, device::Event& iEvent, device::EventSetup const& iSetup) const override {
      [[maybe_unused]] auto const& input = iEvent.get(getToken_);
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add("source", edm::InputTag{});

      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::EDGetToken<portabletest::TestDeviceCollection> getToken_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TestAlpakaGlobalProducerNoOutput);
