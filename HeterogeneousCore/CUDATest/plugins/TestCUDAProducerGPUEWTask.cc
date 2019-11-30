#include "FWCore/Concurrency/interface/FunctorTask.h"
#include "FWCore/Concurrency/interface/WaitingTask.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CUDADataFormats/Common/interface/CUDAProduct.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAContextState.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDATest/interface/CUDAThing.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_noncached_unique_ptr.h"

#include "TestCUDAProducerGPUKernel.h"

#include <thread>

class TestCUDAProducerGPUEWTask : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit TestCUDAProducerGPUEWTask(const edm::ParameterSet& iConfig);
  ~TestCUDAProducerGPUEWTask() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void acquire(const edm::Event& iEvent,
               const edm::EventSetup& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  void addSimpleWork(edm::EventNumber_t eventID, edm::StreamID streamID, CUDAScopedContextTask& ctx);

  std::string label_;
  edm::EDGetTokenT<CUDAProduct<CUDAThing>> srcToken_;
  edm::EDPutTokenT<CUDAProduct<CUDAThing>> dstToken_;
  TestCUDAProducerGPUKernel gpuAlgo_;
  CUDAContextState ctxState_;
  cudautils::device::unique_ptr<float[]> devicePtr_;
  cudautils::host::noncached::unique_ptr<float> hostData_;
};

TestCUDAProducerGPUEWTask::TestCUDAProducerGPUEWTask(const edm::ParameterSet& iConfig)
    : label_{iConfig.getParameter<std::string>("@module_label")},
      srcToken_{consumes<CUDAProduct<CUDAThing>>(iConfig.getParameter<edm::InputTag>("src"))},
      dstToken_{produces<CUDAProduct<CUDAThing>>()} {
  edm::Service<CUDAService> cs;
  if (cs->enabled()) {
    hostData_ = cudautils::make_host_noncached_unique<float>();
  }
}

void TestCUDAProducerGPUEWTask::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag());
  descriptions.addWithDefaultLabel(desc);
}

void TestCUDAProducerGPUEWTask::acquire(const edm::Event& iEvent,
                                        const edm::EventSetup& iSetup,
                                        edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  edm::LogVerbatim("TestCUDAProducerGPUEWTask") << label_ << " TestCUDAProducerGPUEWTask::acquire begin event "
                                                << iEvent.id().event() << " stream " << iEvent.streamID();

  const auto& in = iEvent.get(srcToken_);
  CUDAScopedContextAcquire ctx{in, waitingTaskHolder, ctxState_};

  const CUDAThing& input = ctx.get(in);

  devicePtr_ = gpuAlgo_.runAlgo(label_, input.get(), ctx.stream());
  // Mimick the need to transfer some of the GPU data back to CPU to
  // be used for something within this module, or to be put in the
  // event.
  cudaCheck(
      cudaMemcpyAsync(hostData_.get(), devicePtr_.get() + 10, sizeof(float), cudaMemcpyDeviceToHost, ctx.stream()));
  // Push a task to run addSimpleWork() after the asynchronous work
  // (and acquire()) has finished instead of produce()
  ctx.pushNextTask([iev = iEvent.id().event(), istr = iEvent.streamID(), this](CUDAScopedContextTask ctx) {
    addSimpleWork(iev, istr, ctx);
  });

  edm::LogVerbatim("TestCUDAProducerGPUEWTask") << label_ << " TestCUDAProducerGPUEWTask::acquire end event "
                                                << iEvent.id().event() << " stream " << iEvent.streamID();
}

void TestCUDAProducerGPUEWTask::addSimpleWork(edm::EventNumber_t eventID,
                                              edm::StreamID streamID,
                                              CUDAScopedContextTask& ctx) {
  if (*hostData_ < 13) {
    edm::LogVerbatim("TestCUDAProducerGPUEWTask")
        << label_ << " TestCUDAProducerGPUEWTask::addSimpleWork begin event " << eventID << " stream " << streamID
        << " 10th element " << *hostData_ << " not satisfied, queueing more work";
    cudaCheck(
        cudaMemcpyAsync(hostData_.get(), devicePtr_.get() + 10, sizeof(float), cudaMemcpyDeviceToHost, ctx.stream()));

    ctx.pushNextTask([eventID, streamID, this](CUDAScopedContextTask ctx) { addSimpleWork(eventID, streamID, ctx); });
    gpuAlgo_.runSimpleAlgo(devicePtr_.get(), ctx.stream());
    edm::LogVerbatim("TestCUDAProducerGPUEWTask")
        << label_ << " TestCUDAProducerGPUEWTask::addSimpleWork end event " << eventID << " stream " << streamID;
  } else {
    edm::LogVerbatim("TestCUDAProducerGPUEWTask")
        << label_ << " TestCUDAProducerGPUEWTask::addSimpleWork event " << eventID << " stream " << streamID
        << " 10th element " << *hostData_ << " not queueing more work";
  }
}

void TestCUDAProducerGPUEWTask::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogVerbatim("TestCUDAProducerGPUEWTask")
      << label_ << " TestCUDAProducerGPUEWTask::produce begin event " << iEvent.id().event() << " stream "
      << iEvent.streamID() << " 10th element " << *hostData_;
  if (*hostData_ != 13) {
    throw cms::Exception("Assert") << "Expecting 10th element to be 13, got " << *hostData_;
  }

  CUDAScopedContextProduce ctx{ctxState_};

  ctx.emplace(iEvent, dstToken_, std::move(devicePtr_));

  edm::LogVerbatim("TestCUDAProducerGPUEWTask") << label_ << " TestCUDAProducerGPUEWTask::produce end event "
                                                << iEvent.id().event() << " stream " << iEvent.streamID();
}

DEFINE_FWK_MODULE(TestCUDAProducerGPUEWTask);
