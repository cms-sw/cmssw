#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/Product.h"
#include "DataFormats/PortableTestObjects/interface/TestHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "HeterogeneousCore/AlpakaCore/interface/ScopedContext.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TestAlpakaTranscriber : public edm::stream::EDProducer<edm::ExternalWork> {
  public:
    TestAlpakaTranscriber(edm::ParameterSet const& config)
        : deviceToken_{consumes(config.getParameter<edm::InputTag>("source"))}, hostToken_{produces()} {}

    void acquire(edm::Event const& event, edm::EventSetup const& setup, edm::WaitingTaskWithArenaHolder task) override {
      // create a context reusing the same device and queue as the producer of the input collection
      auto const& input = event.get(deviceToken_);
      cms::alpakatools::ScopedContextAcquire ctx{input, std::move(task)};

      portabletest::TestDeviceCollection const& deviceProduct = ctx.get(input);

      // allocate a host product based on the metadata of the device product
      hostProduct_ = portabletest::TestHostCollection{deviceProduct->metadata().size(), ctx.queue()};

      // FIXME find a way to avoid the copy when the device product is actually a wrapped host prodict

      // copy the content of the device product to the host product
      alpaka::memcpy(ctx.queue(), hostProduct_.buffer(), deviceProduct.const_buffer());

      // do not wait for the asynchronous operation to complete
    }

    void produce(edm::Event& event, edm::EventSetup const&) override {
      // produce() is called once the asynchronous operation has completed, so there is no need for an explicit wait
      event.emplace(hostToken_, std::move(hostProduct_));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("source");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const edm::EDGetTokenT<cms::alpakatools::Product<Queue, portabletest::TestDeviceCollection>> deviceToken_;
    const edm::EDPutTokenT<portabletest::TestHostCollection> hostToken_;

    // hold the output product between acquire() and produce()
    portabletest::TestHostCollection hostProduct_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TestAlpakaTranscriber);
