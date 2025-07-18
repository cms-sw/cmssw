#include "DataFormats/Portable/interface/PortableObject.h"
#include "DataFormats/PortableTestObjects/interface/TestHostObject.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * This class demonstrates a global EDProducer that
   * - produces a host-side EDProduct that is copied to device automatically
   */
  class TestAlpakaGlobalProducerImplicitCopyToDevice : public global::EDProducer<> {
  public:
    TestAlpakaGlobalProducerImplicitCopyToDevice(edm::ParameterSet const& config)
        : EDProducer<>(config), putToken_{produces()}, putTokenInstance_{produces("instance")} {}

    void produce(edm::StreamID, device::Event& iEvent, device::EventSetup const& iSetup) const override {
      portabletest::TestStruct test{6., 14., 15., 52};
      iEvent.emplace(putToken_, cms::alpakatools::host(), test);
      iEvent.emplace(putTokenInstance_, cms::alpakatools::host(), test);
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const edm::EDPutTokenT<portabletest::TestHostObject> putToken_;
    const edm::EDPutTokenT<portabletest::TestHostObject> putTokenInstance_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TestAlpakaGlobalProducerImplicitCopyToDevice);
