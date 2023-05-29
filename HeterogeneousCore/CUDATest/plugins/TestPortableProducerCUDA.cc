#include "CUDADataFormats/Common/interface/Product.h"
#include "CUDADataFormats/PortableTestObjects/interface/TestDeviceCollection.h"
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

#include "TestAlgo.h"

class TestPortableProducerCUDA : public edm::stream::EDProducer<> {
public:
  TestPortableProducerCUDA(edm::ParameterSet const& config)
      : deviceToken_{produces()}, size_{config.getParameter<int32_t>("size")} {}

  void produce(edm::Event& event, edm::EventSetup const&) override {
    // create a context based on the EDM stream number
    cms::cuda::ScopedContextProduce ctx(event.streamID());

    // run the algorithm, potentially asynchronously
    cudatest::TestDeviceCollection deviceProduct{size_, ctx.stream()};
    algo_.fill(deviceProduct, ctx.stream());

    // put the asynchronous product into the event without waiting
    ctx.emplace(event, deviceToken_, std::move(deviceProduct));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<int32_t>("size");
    descriptions.addWithDefaultLabel(desc);
  }

private:
  const edm::EDPutTokenT<cms::cuda::Product<cudatest::TestDeviceCollection>> deviceToken_;
  const int32_t size_;

  // implementation of the algorithm
  cudatest::TestAlgo algo_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TestPortableProducerCUDA);
