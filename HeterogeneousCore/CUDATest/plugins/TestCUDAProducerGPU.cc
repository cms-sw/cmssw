#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "CUDADataFormats/Common/interface/Product.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDATest/interface/Thing.h"

#include "TestCUDAProducerGPUKernel.h"

class TestCUDAProducerGPU : public edm::global::EDProducer<> {
public:
  explicit TestCUDAProducerGPU(const edm::ParameterSet& iConfig);
  ~TestCUDAProducerGPU() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::StreamID streamID, edm::Event& iEvent, edm::EventSetup const& iSetup) const override;

private:
  std::string const label_;
  edm::EDGetTokenT<cms::cuda::Product<cms::cudatest::Thing>> const srcToken_;
  edm::EDPutTokenT<cms::cuda::Product<cms::cudatest::Thing>> const dstToken_;
  TestCUDAProducerGPUKernel const gpuAlgo_;
};

TestCUDAProducerGPU::TestCUDAProducerGPU(edm::ParameterSet const& iConfig)
    : label_(iConfig.getParameter<std::string>("@module_label")),
      srcToken_(consumes<cms::cuda::Product<cms::cudatest::Thing>>(iConfig.getParameter<edm::InputTag>("src"))),
      dstToken_(produces<cms::cuda::Product<cms::cudatest::Thing>>()) {}

void TestCUDAProducerGPU::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag())->setComment("Source of cms::cuda::Product<cms::cudatest::Thing>.");
  descriptions.addWithDefaultLabel(desc);
  descriptions.setComment(
      "This EDProducer is part of the TestCUDAProducer* family. It models a GPU algorithm this is not the first "
      "algorithm in the chain of the GPU EDProducers. Produces cms::cuda::Product<cms::cudatest::Thing>.");
}

void TestCUDAProducerGPU::produce(edm::StreamID streamID, edm::Event& iEvent, edm::EventSetup const& iSetup) const {
  edm::LogVerbatim("TestCUDAProducerGPU") << label_ << " TestCUDAProducerGPU::produce begin event "
                                          << iEvent.id().event() << " stream " << iEvent.streamID();

  auto const& in = iEvent.get(srcToken_);
  cms::cuda::ScopedContextProduce ctx{in};
  cms::cudatest::Thing const& input = ctx.get(in);

  ctx.emplace(iEvent, dstToken_, cms::cudatest::Thing{gpuAlgo_.runAlgo(label_, input.get(), ctx.stream())});

  edm::LogVerbatim("TestCUDAProducerGPU")
      << label_ << " TestCUDAProducerGPU::produce end event " << iEvent.id().event() << " stream " << iEvent.streamID();
}

DEFINE_FWK_MODULE(TestCUDAProducerGPU);
