#include "HeterogeneousCore/Producer/interface/HeterogeneousEDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDACore/interface/GPUCuda.h"
#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"

#include "TestHeterogeneousEDProducerGPUHelpers.h"

#include <chrono>
#include <random>
#include <thread>

#include <cuda.h>
#include <cuda_runtime.h>

/**
 * The purpose of this test is to demonstrate the following
 * - EDProducer implementing an algorithm for CPU and a CUDA GPU
 * - How to initialize the GPU algorithm and make once-per-job-per-stream allocations on the device
 * - How to read heterogeneous product from event
 * - How to write heterogeneous product to event
 *   * Especially pointers to device memory
 */
class TestHeterogeneousEDProducerGPU: public HeterogeneousEDProducer<heterogeneous::HeterogeneousDevices <
                                                                       heterogeneous::GPUCuda,
                                                                       heterogeneous::CPU
                                                                       > > {
public:
  explicit TestHeterogeneousEDProducerGPU(edm::ParameterSet const& iConfig);
  ~TestHeterogeneousEDProducerGPU() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  using OutputType = HeterogeneousProductImpl<heterogeneous::CPUProduct<unsigned int>,
                                              heterogeneous::GPUCudaProduct<TestHeterogeneousEDProducerGPUTask::ResultTypeRaw>>;

  void beginStreamGPUCuda(edm::StreamID streamId, cuda::stream_t<>& cudaStream) override;

  void acquireGPUCuda(const edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup, cuda::stream_t<>& cudaStream) override;

  void produceCPU(edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup) override;
  void produceGPUCuda(edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup, cuda::stream_t<>& cudaStream) override;

  std::string label_;
  edm::EDGetTokenT<HeterogeneousProduct> srcToken_;

  // GPU stuff
  std::unique_ptr<TestHeterogeneousEDProducerGPUTask> gpuAlgo_;
  TestHeterogeneousEDProducerGPUTask::ResultType gpuOutput_;
};

TestHeterogeneousEDProducerGPU::TestHeterogeneousEDProducerGPU(edm::ParameterSet const& iConfig):
  HeterogeneousEDProducer(iConfig),
  label_(iConfig.getParameter<std::string>("@module_label"))
{
  auto srcTag = iConfig.getParameter<edm::InputTag>("src");
  if(!srcTag.label().empty()) {
    srcToken_ = consumesHeterogeneous(srcTag);
  }

  produces<HeterogeneousProduct>();
}

void TestHeterogeneousEDProducerGPU::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag());
  HeterogeneousEDProducer::fillPSetDescription(desc);
  descriptions.add("testHeterogeneousEDProducerGPU", desc);
}

void TestHeterogeneousEDProducerGPU::beginStreamGPUCuda(edm::StreamID streamId, cuda::stream_t<>& cudaStream) {
  edm::Service<CUDAService> cs;

  edm::LogPrint("TestHeterogeneousEDProducerGPU") << " " << label_ << " TestHeterogeneousEDProducerGPU::beginStreamGPUCuda begin stream " << streamId << " device " << cs->getCurrentDevice();

  gpuAlgo_ = std::make_unique<TestHeterogeneousEDProducerGPUTask>();

  edm::LogPrint("TestHeterogeneousEDProducerGPU") << " " << label_ << " TestHeterogeneousEDProducerGPU::beginStreamGPUCuda end stream " << streamId << " device " << cs->getCurrentDevice();
}

void TestHeterogeneousEDProducerGPU::acquireGPUCuda(const edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup, cuda::stream_t<>& cudaStream) {
  edm::Service<CUDAService> cs;
  edm::LogPrint("TestHeterogeneousEDProducerGPU") << " " << label_ << " TestHeterogeneousEDProducerGPU::acquireGPUCuda begin event " << iEvent.id().event() << " stream " << iEvent.streamID() << " device " << cs->getCurrentDevice();

  gpuOutput_.first.reset();
  gpuOutput_.second.reset();

  TestHeterogeneousEDProducerGPUTask::ResultTypeRaw input = std::make_pair(nullptr, nullptr);
  if(!srcToken_.isUninitialized()) {
    edm::Handle<TestHeterogeneousEDProducerGPUTask::ResultTypeRaw> hin;
    iEvent.getByToken<OutputType>(srcToken_, hin);
    input = *hin;
  }

  gpuOutput_ = gpuAlgo_->runAlgo(label_, 0, input, cudaStream);

  edm::LogPrint("TestHeterogeneousEDProducerGPU") << " " << label_ << " TestHeterogeneousEDProducerGPU::acquireGPUCuda end event " << iEvent.id().event() << " stream " << iEvent.streamID() << " device " << cs->getCurrentDevice();
}

void TestHeterogeneousEDProducerGPU::produceCPU(edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup) {
  edm::LogPrint("TestHeterogeneousEDProducerGPU") << label_ << " TestHeterogeneousEDProducerGPU::produceCPU begin event " << iEvent.id().event() << " stream " << iEvent.streamID();

  unsigned int input = 0;
  if(!srcToken_.isUninitialized()) {
    edm::Handle<unsigned int> hin;
    iEvent.getByToken<OutputType>(srcToken_, hin);
    input = *hin;
  }

  std::random_device r;
  std::mt19937 gen(r());
  auto dist = std::uniform_real_distribution<>(1.0, 3.0); 
  auto dur = dist(gen);
  edm::LogPrint("TestHeterogeneousEDProducerGPU") << "  Task (CPU) for event " << iEvent.id().event() << " in stream " << iEvent.streamID() << " will take " << dur << " seconds";
  std::this_thread::sleep_for(std::chrono::seconds(1)*dur);

  const unsigned int output = input + iEvent.streamID()*100 + iEvent.id().event();

  iEvent.put<OutputType>(std::make_unique<unsigned int>(output));

  edm::LogPrint("TestHeterogeneousEDProducerGPU") << label_ << " TestHeterogeneousEDProducerGPU::produceCPU end event " << iEvent.id().event() << " stream " << iEvent.streamID() << " result " << output;
}

void TestHeterogeneousEDProducerGPU::produceGPUCuda(edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup, cuda::stream_t<>& cudaStream) {
  edm::Service<CUDAService> cs;
  edm::LogPrint("TestHeterogeneousEDProducerGPU") << label_ << " TestHeterogeneousEDProducerGPU::produceGPUCuda begin event " << iEvent.id().event() << " stream " << iEvent.streamID() << " device " << cs->getCurrentDevice();

  gpuAlgo_->release(label_, cudaStream);
  iEvent.put<OutputType>(std::make_unique<TestHeterogeneousEDProducerGPUTask::ResultTypeRaw>(gpuOutput_.first.get(), gpuOutput_.second.get()),
                         [this, eventId=iEvent.event().id().event(), streamId=iEvent.event().streamID(),
                          dev=cs->getCurrentDevice(), &cudaStream
                          ](const TestHeterogeneousEDProducerGPUTask::ResultTypeRaw& src, unsigned int& dst) {
                           // TODO: try to abstract both the current device setting and the delivery of cuda::stream to this function
                           // It needs some further thought so I leave it now as it is
                           // Maybe "per-thread default stream" would help as they are regular CUDA streams (wrt. to the default stream)?
                           // Or not, because the current device has to be set correctly.
                           // Maybe we should initiate the transfer in all cases?
                           cuda::device::current::scoped_override_t<> setDeviceForThisScope(dev);
                           edm::LogPrint("TestHeterogeneousEDProducerGPU") << "  " << label_ << " Copying from GPU to CPU for event " << eventId << " in stream " << streamId;
                           dst = TestHeterogeneousEDProducerGPUTask::getResult(src, cudaStream);
                         });

  // If, for any reason, you want to disable the automatic GPU->CPU transfer, pass heterogeneous::DisableTransfer{} insteads of the function, i.e.
  //iEvent.put<OutputType>(std::make_unique<TestHeterogeneousEDProducerGPUTask::ResultTypeRaw>(gpuOutput_.first.get(), gpuOutput_.second.get()), heterogeneous::DisableTransfer{});

  edm::LogPrint("TestHeterogeneousEDProducerGPU") << label_ << " TestHeterogeneousEDProducerGPU::produceGPUCuda end event " << iEvent.id().event() << " stream " << iEvent.streamID() << " device " << cs->getCurrentDevice();
}

DEFINE_FWK_MODULE(TestHeterogeneousEDProducerGPU);
