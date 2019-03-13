#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "CUDADataFormats/Common/interface/CUDAProduct.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAContextToken.h"
#include "HeterogeneousCore/CUDATest/interface/CUDAThing.h"

#include "TestCUDAProducerGPUKernel.h"

class TestCUDAProducerGPUEW: public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit TestCUDAProducerGPUEW(const edm::ParameterSet& iConfig);
  ~TestCUDAProducerGPUEW() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
private:
  std::string label_;
  edm::EDGetTokenT<CUDAProduct<CUDAThing>> srcToken_;
  edm::EDPutTokenT<CUDAProduct<CUDAThing>> dstToken_;
  TestCUDAProducerGPUKernel gpuAlgo_;
  CUDAContextToken ctxTmp_;
  cudautils::device::unique_ptr<float[]> devicePtr_;
  float hostData_ = 0.f;
};

TestCUDAProducerGPUEW::TestCUDAProducerGPUEW(const edm::ParameterSet& iConfig):
  label_{iConfig.getParameter<std::string>("@module_label")},
  srcToken_{consumes<CUDAProduct<CUDAThing>>(iConfig.getParameter<edm::InputTag>("src"))},
  dstToken_{produces<CUDAProduct<CUDAThing>>()}
{}

void TestCUDAProducerGPUEW::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag());
  descriptions.addWithDefaultLabel(desc);
}

void TestCUDAProducerGPUEW::acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  edm::LogVerbatim("TestCUDAProducerGPUEW") << label_ << " TestCUDAProducerGPUEW::acquire begin event " << iEvent.id().event() << " stream " << iEvent.streamID();

  const auto& in = iEvent.get(srcToken_);
  CUDAScopedContext ctx{in, std::move(waitingTaskHolder)};
  const CUDAThing& input = ctx.get(in);

  devicePtr_ = gpuAlgo_.runAlgo(label_, input.get(), ctx.stream());
  // Mimick the need to transfer some of the GPU data back to CPU to
  // be used for something within this module, or to be put in the
  // event.
  cuda::memory::async::copy(&hostData_, devicePtr_.get()+10, sizeof(float), ctx.stream().id());

  edm::LogVerbatim("TestCUDAProducerGPUEW") << label_ << " TestCUDAProducerGPUEW::acquire end event " << iEvent.id().event() << " stream " << iEvent.streamID();

  ctxTmp_ = ctx.toToken();
}

void TestCUDAProducerGPUEW::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogVerbatim("TestCUDAProducerGPUEW") << label_ << " TestCUDAProducerGPUEW::produce begin event " << iEvent.id().event() << " stream " << iEvent.streamID() << " 10th element " << hostData_; 

  CUDAScopedContext ctx{std::move(ctxTmp_)};

  ctx.emplace(iEvent, dstToken_, std::move(devicePtr_));

  edm::LogVerbatim("TestCUDAProducerGPUEW") << label_ << " TestCUDAProducerGPUEW::produce end event " << iEvent.id().event() << " stream " << iEvent.streamID();
}

DEFINE_FWK_MODULE(TestCUDAProducerGPUEW);
