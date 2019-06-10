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

class TestCUDAProducerGPUFirst: public edm::global::EDProducer<> {
public:
  explicit TestCUDAProducerGPUFirst(const edm::ParameterSet& iConfig);
  ~TestCUDAProducerGPUFirst() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::StreamID stream, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;
private:
  std::string label_;
  TestCUDAProducerGPUKernel gpuAlgo_;
};

TestCUDAProducerGPUFirst::TestCUDAProducerGPUFirst(const edm::ParameterSet& iConfig):
  label_(iConfig.getParameter<std::string>("@module_label"))
{
  produces<CUDAProduct<CUDAThing>>();
}

void TestCUDAProducerGPUFirst::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.addWithDefaultLabel(desc);
  descriptions.setComment("This EDProducer is part of the TestCUDAProducer* family. It models a GPU algorithm this the first algorithm in the chain of the GPU EDProducers. Produces CUDA<ProductCUDAThing>.");
}

void TestCUDAProducerGPUFirst::produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::LogVerbatim("TestCUDAProducerGPUFirst") << label_ << " TestCUDAProducerGPUFirst::produce begin event " << iEvent.id().event() << " stream " << iEvent.streamID();

  CUDAScopedContext ctx{streamID};

  cudautils::device::unique_ptr<float[]> output = gpuAlgo_.runAlgo(label_, ctx.stream());
  iEvent.put(ctx.wrap(CUDAThing(std::move(output))));

  edm::LogVerbatim("TestCUDAProducerGPUFirst") << label_ << " TestCUDAProducerGPUFirst::produce end event " << iEvent.id().event() << " stream " << iEvent.streamID();
}

DEFINE_FWK_MODULE(TestCUDAProducerGPUFirst);
