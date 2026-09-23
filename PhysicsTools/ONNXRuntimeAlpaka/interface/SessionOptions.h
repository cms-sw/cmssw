#ifndef PhysicsTools_ONNXRuntimeAlpaka_interface_SessionOptions_h
#define PhysicsTools_ONNXRuntimeAlpaka_interface_SessionOptions_h

#include <string>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "onnxruntime/onnxruntime_cxx_api.h"

#include "PhysicsTools/ONNXRuntimeAlpaka/interface/OrtEnvironment.h"

namespace cms::Ort::alpakatools {

  // Build the ONNX Runtime session options to run the inference on the device and in the queue given.
  //
  // This plays the role of the QueueGuard in PyTorchAlpaka: ONNX Runtime cannot switch to a different stream at each
  // inference call, so the stream underlying the alpaka queue must be bound to the session when it is created.
  // All the kernels and memory operations of the session are then submitted to that stream, asynchronously.
  template <typename TQueue>
    requires(alpaka::isQueue<TQueue>)::Ort::SessionOptions
  sessionOptions(TQueue const& queue) {
    // The ONNX Runtime environment must exist before the execution providers are configured, because it sets up the
    // default logger used by the provider libraries.
    cms::Ort::alpakatools::environment();

    ::Ort::SessionOptions options;
    // Disable the ONNX Runtime internal threading model: all CPU based operations run single-threaded, in the
    // calling thread, like the PyTorchService does for PyTorch.
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    if constexpr (std::is_same_v<alpaka::Dev<TQueue>, alpaka::DevCudaRt>) {
      ::Ort::CUDAProviderOptions cuda;
      cuda.Update({
          {"device_id", std::to_string(alpaka::getDev(queue).getNativeHandle())},
          // Run the memory copies in the compute stream: alpaka queues are non-blocking, so they would not be
          // synchronised with the legacy default stream.
          {"do_copy_in_default_stream", "0"},
          // Grow the memory arena only by the amount requested.
          {"arena_extend_strategy", "kSameAsRequested"},
          // Do not benchmark the convolution algorithms for each new input shape.
          {"cudnn_conv_algo_search", "HEURISTIC"},
          // Use full FP32 precision, like PyTorch does by default for matrix multiplications.
          {"use_tf32", "0"},
      });
      // Note: a user compute stream cannot be combined with an external allocator ("gpu_external_alloc").
      cuda.UpdateWithValue("user_compute_stream", queue.getNativeHandle());
      options.AppendExecutionProvider_CUDA_V2(*cuda);
    }
#endif

    return options;
  }

}  // namespace cms::Ort::alpakatools

#endif  // PhysicsTools_ONNXRuntimeAlpaka_interface_SessionOptions_h
