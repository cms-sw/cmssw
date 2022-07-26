// The "Transcriber" makes sense only across different memory spaces
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) or defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#include <optional>
#include <string>

#include <alpaka/alpaka.hpp>

#include "DataFormats/PortableTestObjects/interface/TestHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaServices/interface/alpaka/AlpakaService.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TestAlpakaTranscriber : public edm::stream::EDProducer<> {
  public:
    TestAlpakaTranscriber(edm::ParameterSet const& config)
        : deviceToken_{consumes(config.getParameter<edm::InputTag>("source"))}, hostToken_{produces()} {}

    void beginStream(edm::StreamID sid) override {
      // choose a device based on the EDM stream number
      edm::Service<ALPAKA_TYPE_ALIAS(AlpakaService)> service;
      if (not service->enabled()) {
        throw cms::Exception("Configuration") << ALPAKA_TYPE_ALIAS_NAME(AlpakaService) << " is disabled.";
      }
      auto& devices = service->devices();
      unsigned int index = sid.value() % devices.size();
      device_ = devices[index];
    }

    void produce(edm::Event& event, edm::EventSetup const&) override {
      // create a queue to submit async work
      Queue queue{*device_};
      portabletest::TestDeviceCollection const& deviceProduct = event.get(deviceToken_);

      portabletest::TestHostCollection hostProduct{deviceProduct->metadata().size(), alpaka_common::host(), *device_};
      alpaka::memcpy(queue, hostProduct.buffer(), deviceProduct.const_buffer());

      // wait for any async work to complete
      alpaka::wait(queue);

      event.emplace(hostToken_, std::move(hostProduct));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("source");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const edm::EDGetTokenT<portabletest::TestDeviceCollection> deviceToken_;
    const edm::EDPutTokenT<portabletest::TestHostCollection> hostToken_;

    // device associated to the EDM stream
    std::optional<Device> device_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TestAlpakaTranscriber);

#endif  // defined(ALPAKA_ACC_GPU_CUDA_ENABLED) or defined(ALPAKA_ACC_GPU_HIP_ENABLED)
