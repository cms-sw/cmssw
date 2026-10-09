#ifndef PhysicsTools_PyTorchAlpaka_interface_alpaka_PyTorchEDProducer_h
#define PhysicsTools_PyTorchAlpaka_interface_alpaka_PyTorchEDProducer_h

#include <exception>

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/FixedQueueEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/FixedQueueRegistry.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/PyTorchAllocatorBridge.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::stream {

  template <typename... Args>
  class PyTorchEDProducer : public FixedQueueEDProducer<Args...> {
    using Base = FixedQueueEDProducer<Args...>;

  protected:
    explicit PyTorchEDProducer(edm::ParameterSet const& config) : Base(config) {}

    // Optional hooks for derived producers.
    //
    // beginStreamHook() is called after the queue has been registered
    // with the PyTorch allocator bridge.
    virtual void beginStreamHook(edm::StreamID, Queue) {}

    // endStreamHook() is called before the BLAS workspace is cleared
    // and before the queue is unregistered.
    virtual void endStreamHook(Queue) {}

  public:
    void beginStream(edm::StreamID sid, Queue queue) final {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
      if (PyTorchAllocatorBridge::isActive())
        cms::alpakatools::getFixedQueueRegistry<Queue>().registerQueue(queue);
#else
      beginStreamHook(sid, queue);
#endif
    }

    void endStream(Queue queue) final {
      endStreamHook(queue);
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
      if (PyTorchAllocatorBridge::isActive()) {
        PyTorchAllocatorBridge::clearBlasWorkspace(queue);
        cms::alpakatools::getFixedQueueRegistry<Queue>().unregisterQueue(queue);
      }
#endif
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::stream

#endif  // PhysicsTools_PyTorchAlpaka_interface_alpaka_PyTorchEDProducer_h
