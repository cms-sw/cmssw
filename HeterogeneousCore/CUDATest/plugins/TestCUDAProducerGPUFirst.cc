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

class TestCUDAProducerGPUFirst : public edm::global::EDProducer<> {
public:
  explicit TestCUDAProducerGPUFirst(edm::ParameterSet const& iConfig);
  ~TestCUDAProducerGPUFirst() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::StreamID stream, edm::Event& iEvent, edm::EventSetup const& iSetup) const override;

private:
  std::string const label_;
  edm::EDPutTokenT<cms::cuda::Product<cms::cudatest::Thing>> const dstToken_;
  TestCUDAProducerGPUKernel const gpuAlgo_;
};

TestCUDAProducerGPUFirst::TestCUDAProducerGPUFirst(edm::ParameterSet const& iConfig)
    : label_(iConfig.getParameter<std::string>("@module_label")),
      dstToken_{produces<cms::cuda::Product<cms::cudatest::Thing>>()} {}

void TestCUDAProducerGPUFirst::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.addWithDefaultLabel(desc);
  descriptions.setComment(
      "This EDProducer is part of the TestCUDAProducer* family. It models a GPU algorithm this the first algorithm in "
      "the chain of the GPU EDProducers. Produces cms::cuda::Productcms::cudatest::Thing>.");
}

void TestCUDAProducerGPUFirst::produce(edm::StreamID streamID,
                                       edm::Event& iEvent,
                                       edm::EventSetup const& iSetup) const {
  edm::LogVerbatim("TestCUDAProducerGPUFirst") << label_ << " TestCUDAProducerGPUFirst::produce begin event "
                                               << iEvent.id().event() << " stream " << iEvent.streamID();

  cms::cuda::ScopedContextProduce ctx{streamID};

  cms::cuda::device::unique_ptr<float[]> output = gpuAlgo_.runAlgo(label_, ctx.stream());
  ctx.emplace(iEvent, dstToken_, std::move(output));

  edm::LogVerbatim("TestCUDAProducerGPUFirst") << label_ << " TestCUDAProducerGPUFirst::produce end event "
                                               << iEvent.id().event() << " stream " << iEvent.streamID();
}

DEFINE_FWK_MODULE(TestCUDAProducerGPUFirst);
