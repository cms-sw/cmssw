#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "CUDADataFormats/Common/interface/CUDAProduct.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
#include "HeterogeneousCore/CUDATest/interface/CUDAThing.h"

#include "TestCUDAProducerGPUKernel.h"

class TestCUDAProducerGPU: public edm::global::EDProducer<> {
public:
  explicit TestCUDAProducerGPU(const edm::ParameterSet& iConfig);
  ~TestCUDAProducerGPU() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;
private:
  std::string label_;
  edm::EDGetTokenT<CUDAProduct<CUDAThing>> srcToken_;
  edm::EDPutTokenT<CUDAProduct<CUDAThing>> dstToken_;
  TestCUDAProducerGPUKernel gpuAlgo_;
};

TestCUDAProducerGPU::TestCUDAProducerGPU(const edm::ParameterSet& iConfig):
  label_(iConfig.getParameter<std::string>("@module_label")),
  srcToken_(consumes<CUDAProduct<CUDAThing>>(iConfig.getParameter<edm::InputTag>("src"))),
  dstToken_(produces<CUDAProduct<CUDAThing>>())
{}

void TestCUDAProducerGPU::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag())->setComment("Source of CUDAProduct<CUDAThing>.");
  descriptions.addWithDefaultLabel(desc);
  descriptions.setComment("This EDProducer is part of the TestCUDAProducer* family. It models a GPU algorithm this is not the first algorithm in the chain of the GPU EDProducers. Produces CUDAProduct<CUDAThing>.");
}

void TestCUDAProducerGPU::produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::LogVerbatim("TestCUDAProducerGPU") << label_ << " TestCUDAProducerGPU::produce begin event " << iEvent.id().event() << " stream " << iEvent.streamID();

  const auto& in = iEvent.get(srcToken_);
  CUDAScopedContext ctx{in};
  const CUDAThing& input = ctx.get(in);

  ctx.emplace(iEvent, dstToken_, CUDAThing{gpuAlgo_.runAlgo(label_, input.get(), ctx.stream())});

  edm::LogVerbatim("TestCUDAProducerGPU") << label_ << " TestCUDAProducerGPU::produce end event " << iEvent.id().event() << " stream " << iEvent.streamID();
}

DEFINE_FWK_MODULE(TestCUDAProducerGPU);
