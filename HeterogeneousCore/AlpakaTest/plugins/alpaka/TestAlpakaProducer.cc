#include <optional>
#include <string>

#include <alpaka/alpaka.hpp>

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
#include "HeterogeneousCore/AlpakaServices/interface/alpaka/AlpakaService.h"

#include "TestAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TestAlpakaProducer : public edm::stream::EDProducer<> {
  public:
    TestAlpakaProducer(edm::ParameterSet const& config)
        : deviceToken_{produces()}, size_{config.getParameter<int32_t>("size")} {}

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
      portabletest::TestDeviceCollection deviceProduct{size_, *device_};

      // run the algorithm, potentially asynchronously
      algo_.fill(queue, deviceProduct);

      // wait for any asynchronous work to complete
      alpaka::wait(queue);

      event.emplace(deviceToken_, std::move(deviceProduct));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<int32_t>("size");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const edm::EDPutTokenT<portabletest::TestDeviceCollection> deviceToken_;
    const int32_t size_;

    // device associated to the EDM stream
    std::optional<Device> device_;

    // implementation of the algorithm
    TestAlgo algo_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TestAlpakaProducer);
