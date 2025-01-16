#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceObject.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "verifyDeviceObjectAsync.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class TestAlpakaVerifyObjectOnDevice : public stream::SynchronizingEDProducer<> {
  public:
    TestAlpakaVerifyObjectOnDevice(edm::ParameterSet const& config)
        : SynchronizingEDProducer<>(config),
          getToken_{consumes(config.getParameter<edm::InputTag>("source"))},
          putToken_{produces()} {}

    void acquire(device::Event const& iEvent, device::EventSetup const& iSetup) override {
      auto const& deviceObject = iEvent.get(getToken_);
      succeeded_ = verifyDeviceObjectAsync(iEvent.queue(), deviceObject);
    }

    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override {
      if (not **succeeded_) {
        throw cms::Exception("Assert") << "Device object verification failed";
      }
      iEvent.emplace(putToken_, true);
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("source");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::EDGetToken<portabletest::TestDeviceObject> getToken_;
    const edm::EDPutTokenT<bool> putToken_;
    std::optional<cms::alpakatools::host_buffer<bool>> succeeded_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TestAlpakaVerifyObjectOnDevice);
