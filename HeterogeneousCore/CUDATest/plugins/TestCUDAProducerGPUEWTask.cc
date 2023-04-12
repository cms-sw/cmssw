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

#include "CUDADataFormats/Common/interface/Product.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDACore/interface/ContextState.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAInterface.h"
#include "HeterogeneousCore/CUDATest/interface/Thing.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_noncached_unique_ptr.h"

#include "TestCUDAProducerGPUKernel.h"

#include <thread>

class TestCUDAProducerGPUEWTask : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit TestCUDAProducerGPUEWTask(edm::ParameterSet const& iConfig);
  ~TestCUDAProducerGPUEWTask() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  void addSimpleWork(edm::EventNumber_t eventID, edm::StreamID streamID, cms::cuda::ScopedContextTask& ctx);

  std::string const label_;
  edm::EDGetTokenT<cms::cuda::Product<cms::cudatest::Thing>> const srcToken_;
  edm::EDPutTokenT<cms::cuda::Product<cms::cudatest::Thing>> const dstToken_;
  TestCUDAProducerGPUKernel gpuAlgo_;
  cms::cuda::ContextState ctxState_;
  cms::cuda::device::unique_ptr<float[]> devicePtr_;
  cms::cuda::host::noncached::unique_ptr<float> hostData_;
};

TestCUDAProducerGPUEWTask::TestCUDAProducerGPUEWTask(edm::ParameterSet const& iConfig)
    : label_{iConfig.getParameter<std::string>("@module_label")},
      srcToken_{consumes<cms::cuda::Product<cms::cudatest::Thing>>(iConfig.getParameter<edm::InputTag>("src"))},
      dstToken_{produces<cms::cuda::Product<cms::cudatest::Thing>>()} {
  edm::Service<CUDAInterface> cuda;
  if (cuda and cuda->enabled()) {
    hostData_ = cms::cuda::make_host_noncached_unique<float>();
  }
}

void TestCUDAProducerGPUEWTask::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag());
  descriptions.addWithDefaultLabel(desc);
  descriptions.setComment(
      "This EDProducer is part of the TestCUDAProducer* family. It models a GPU algorithm this is not the first "
      "algorithm in the chain of the GPU EDProducers, and that transfers some data from GPU to CPU multiple times "
      "alternating the transfers and kernel executions (e.g. to decide which kernel to run next based on a value from "
      "GPU). A synchronization between GPU and CPU is needed after each transfer. The synchronizations are implemented "
      "with the ExternalWork extension and explicit TBB tasks within the module. Produces "
      "cms::cuda::Product<cms::cudatest::Thing>.");
}

void TestCUDAProducerGPUEWTask::acquire(edm::Event const& iEvent,
                                        edm::EventSetup const& iSetup,
                                        edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  edm::LogVerbatim("TestCUDAProducerGPUEWTask") << label_ << " TestCUDAProducerGPUEWTask::acquire begin event "
                                                << iEvent.id().event() << " stream " << iEvent.streamID();

  auto const& in = iEvent.get(srcToken_);
  cms::cuda::ScopedContextAcquire ctx{in, waitingTaskHolder, ctxState_};

  cms::cudatest::Thing const& input = ctx.get(in);

  devicePtr_ = gpuAlgo_.runAlgo(label_, input.get(), ctx.stream());
  // Mimick the need to transfer some of the GPU data back to CPU to
  // be used for something within this module, or to be put in the
  // event.
  cudaCheck(
      cudaMemcpyAsync(hostData_.get(), devicePtr_.get() + 10, sizeof(float), cudaMemcpyDeviceToHost, ctx.stream()));
  // Push a task to run addSimpleWork() after the asynchronous work
  // (and acquire()) has finished instead of produce()
  ctx.pushNextTask([iev = iEvent.id().event(), istr = iEvent.streamID(), this](cms::cuda::ScopedContextTask ctx) {
    addSimpleWork(iev, istr, ctx);
  });

  edm::LogVerbatim("TestCUDAProducerGPUEWTask") << label_ << " TestCUDAProducerGPUEWTask::acquire end event "
                                                << iEvent.id().event() << " stream " << iEvent.streamID();
}

void TestCUDAProducerGPUEWTask::addSimpleWork(edm::EventNumber_t eventID,
                                              edm::StreamID streamID,
                                              cms::cuda::ScopedContextTask& ctx) {
  if (*hostData_ < 13) {
    edm::LogVerbatim("TestCUDAProducerGPUEWTask")
        << label_ << " TestCUDAProducerGPUEWTask::addSimpleWork begin event " << eventID << " stream " << streamID
        << " 10th element " << *hostData_ << " not satisfied, queueing more work";
    cudaCheck(
        cudaMemcpyAsync(hostData_.get(), devicePtr_.get() + 10, sizeof(float), cudaMemcpyDeviceToHost, ctx.stream()));

    ctx.pushNextTask(
        [eventID, streamID, this](cms::cuda::ScopedContextTask ctx) { addSimpleWork(eventID, streamID, ctx); });
    gpuAlgo_.runSimpleAlgo(devicePtr_.get(), ctx.stream());
    edm::LogVerbatim("TestCUDAProducerGPUEWTask")
        << label_ << " TestCUDAProducerGPUEWTask::addSimpleWork end event " << eventID << " stream " << streamID;
  } else {
    edm::LogVerbatim("TestCUDAProducerGPUEWTask")
        << label_ << " TestCUDAProducerGPUEWTask::addSimpleWork event " << eventID << " stream " << streamID
        << " 10th element " << *hostData_ << " not queueing more work";
  }
}

void TestCUDAProducerGPUEWTask::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  edm::LogVerbatim("TestCUDAProducerGPUEWTask")
      << label_ << " TestCUDAProducerGPUEWTask::produce begin event " << iEvent.id().event() << " stream "
      << iEvent.streamID() << " 10th element " << *hostData_;
  if (*hostData_ != 13) {
    throw cms::Exception("Assert") << "Expecting 10th element to be 13, got " << *hostData_;
  }

  cms::cuda::ScopedContextProduce ctx{ctxState_};

  ctx.emplace(iEvent, dstToken_, std::move(devicePtr_));

  edm::LogVerbatim("TestCUDAProducerGPUEWTask") << label_ << " TestCUDAProducerGPUEWTask::produce end event "
                                                << iEvent.id().event() << " stream " << iEvent.streamID();
}

DEFINE_FWK_MODULE(TestCUDAProducerGPUEWTask);
