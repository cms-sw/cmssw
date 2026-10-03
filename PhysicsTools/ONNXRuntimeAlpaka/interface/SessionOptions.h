#ifndef PhysicsTools_ONNXRuntimeAlpaka_interface_SessionOptions_h
#define PhysicsTools_ONNXRuntimeAlpaka_interface_SessionOptions_h

#include <type_traits>

#include <alpaka/alpaka.hpp>

#include <onnxruntime/onnxruntime_cxx_api.h>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/ONNXRuntime/interface/Backend.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXInterface.h"

namespace cms::Ort::alpakatools {

  // Build the ONNX Runtime session options to run the inference on the device and in the queue given.
  //
  // This plays the role of the QueueGuard in PyTorchAlpaka: ONNX Runtime cannot switch to a different stream at each
  // inference call, so the stream underlying the alpaka queue must be bound to the session when it is created, using
  // the "user_compute_stream" option of the CUDA or MIGraphX execution provider.
  // All the kernels and memory operations of the session are then submitted to that stream, asynchronously.
  template <typename TQueue>
    requires(alpaka::isQueue<TQueue>)::Ort::SessionOptions
  sessionOptions(TQueue const& queue) {
    edm::Service<cms::Ort::ONNXInterface> onnx;

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    if constexpr (std::is_same_v<alpaka::Dev<TQueue>, alpaka::DevCudaRt>) {
      // Run the session on the device of the queue, and in the stream underlying it: the device is chosen by the
      // framework, which already distributes the framework streams over the devices.
      return onnx->sessionOptions(
          cms::Ort::Backend::cuda, alpaka::getDev(queue).getNativeHandle(), queue.getNativeHandle());
    }
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    if constexpr (std::is_same_v<alpaka::Dev<TQueue>, alpaka::DevHipRt>) {
      // Run the session on the device of the queue, and in the stream underlying it, like for the CUDA backend.
      return onnx->sessionOptions(
          cms::Ort::Backend::rocm, alpaka::getDev(queue).getNativeHandle(), queue.getNativeHandle());
    }
#endif

    // The serial_sync backend runs the session on the CPU.
    return onnx->sessionOptions(cms::Ort::Backend::cpu);
  }

}  // namespace cms::Ort::alpakatools

#endif  // PhysicsTools_ONNXRuntimeAlpaka_interface_SessionOptions_h
