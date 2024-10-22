#include "CUDADataFormats/Common/interface/Product.h"
#include "CUDADataFormats/PortableTestObjects/interface/TestDeviceCollection.h"
#include "CUDADataFormats/PortableTestObjects/interface/TestHostCollection.h"
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
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

class TestPortableTranscriber : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  TestPortableTranscriber(edm::ParameterSet const& config)
      : deviceToken_{consumes(config.getParameter<edm::InputTag>("source"))}, hostToken_{produces()} {}

  void acquire(edm::Event const& event, edm::EventSetup const& setup, edm::WaitingTaskWithArenaHolder task) override {
    // create a context reusing the same device and queue as the producer of the input collection
    auto const& input = event.get(deviceToken_);
    cms::cuda::ScopedContextAcquire ctx{input, std::move(task)};

    cudatest::TestDeviceCollection const& deviceProduct = ctx.get(input);

    // allocate a host product based on the metadata of the device product
    hostProduct_ = cudatest::TestHostCollection{deviceProduct->metadata().size(), ctx.stream()};

    // copy the content of the device product to the host product
    cms::cuda::copyAsync(hostProduct_.buffer(), deviceProduct.const_buffer(), deviceProduct.bufferSize(), ctx.stream());

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
  const edm::EDGetTokenT<cms::cuda::Product<cudatest::TestDeviceCollection>> deviceToken_;
  const edm::EDPutTokenT<cudatest::TestHostCollection> hostToken_;

  // hold the output product between acquire() and produce()
  cudatest::TestHostCollection hostProduct_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TestPortableTranscriber);
