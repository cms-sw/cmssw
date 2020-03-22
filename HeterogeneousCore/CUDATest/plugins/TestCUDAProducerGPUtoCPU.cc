#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "CUDADataFormats/Common/interface/Product.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDATest/interface/Thing.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

#include "TestCUDAProducerGPUKernel.h"

class TestCUDAProducerGPUtoCPU : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit TestCUDAProducerGPUtoCPU(edm::ParameterSet const& iConfig);
  ~TestCUDAProducerGPUtoCPU() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  std::string const label_;
  edm::EDGetTokenT<cms::cuda::Product<cms::cudatest::Thing>> const srcToken_;
  edm::EDPutTokenT<int> const dstToken_;
  cms::cuda::host::unique_ptr<float[]> buffer_;
};

TestCUDAProducerGPUtoCPU::TestCUDAProducerGPUtoCPU(edm::ParameterSet const& iConfig)
    : label_{iConfig.getParameter<std::string>("@module_label")},
      srcToken_{consumes<cms::cuda::Product<cms::cudatest::Thing>>(iConfig.getParameter<edm::InputTag>("src"))},
      dstToken_{produces<int>()} {}

void TestCUDAProducerGPUtoCPU::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag())->setComment("Source for cms::cuda::Product<cms::cudatest::Thing>.");
  descriptions.addWithDefaultLabel(desc);
  descriptions.setComment(
      "This EDProducer is part of the TestCUDAProducer* family. It models the GPU->CPU data transfer and formatting of "
      "the data to legacy data format. Produces int, to be compatible with TestCUDAProducerCPU.");
}

void TestCUDAProducerGPUtoCPU::acquire(edm::Event const& iEvent,
                                       edm::EventSetup const& iSetup,
                                       edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  edm::LogVerbatim("TestCUDAProducerGPUtoCPU") << label_ << " TestCUDAProducerGPUtoCPU::acquire begin event "
                                               << iEvent.id().event() << " stream " << iEvent.streamID();

  auto const& in = iEvent.get(srcToken_);
  cms::cuda::ScopedContextAcquire ctx{in, std::move(waitingTaskHolder)};
  cms::cudatest::Thing const& device = ctx.get(in);

  buffer_ = cms::cuda::make_host_unique<float[]>(TestCUDAProducerGPUKernel::NUM_VALUES, ctx.stream());
  // Enqueue async copy, continue in produce once finished
  cudaCheck(cudaMemcpyAsync(buffer_.get(),
                            device.get(),
                            TestCUDAProducerGPUKernel::NUM_VALUES * sizeof(float),
                            cudaMemcpyDeviceToHost,
                            ctx.stream()));

  edm::LogVerbatim("TestCUDAProducerGPUtoCPU") << label_ << " TestCUDAProducerGPUtoCPU::acquire end event "
                                               << iEvent.id().event() << " stream " << iEvent.streamID();
}

void TestCUDAProducerGPUtoCPU::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  edm::LogVerbatim("TestCUDAProducerGPUtoCPU") << label_ << " TestCUDAProducerGPUtoCPU::produce begin event "
                                               << iEvent.id().event() << " stream " << iEvent.streamID();

  int counter = 0;
  for (int i = 0; i < TestCUDAProducerGPUKernel::NUM_VALUES; ++i) {
    counter += buffer_[i];
  }
  buffer_.reset();  // not so nice, but no way around?

  iEvent.emplace(dstToken_, counter);

  edm::LogVerbatim("TestCUDAProducerGPUtoCPU")
      << label_ << " TestCUDAProducerGPUtoCPU::produce end event " << iEvent.id().event() << " stream "
      << iEvent.streamID() << " result " << counter;
}

DEFINE_FWK_MODULE(TestCUDAProducerGPUtoCPU);
