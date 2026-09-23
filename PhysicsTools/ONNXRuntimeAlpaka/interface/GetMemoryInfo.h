#ifndef PhysicsTools_ONNXRuntimeAlpaka_interface_GetMemoryInfo_h
#define PhysicsTools_ONNXRuntimeAlpaka_interface_GetMemoryInfo_h

#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "onnxruntime/onnxruntime_cxx_api.h"

namespace cms::Ort::alpakatools {

  // Returns true if ONNX Runtime can run directly on the given alpaka device.
  // ONNX Runtime in CMSSW is built with the CPU and CUDA execution providers only, so on AMD GPUs the inference falls
  // back to the CPU, copying the tensors to the host and back.
  template <typename TDev>
    requires(alpaka::isDevice<TDev>)
  constexpr bool isNativeDevice() {
    if constexpr (std::is_same_v<alpaka::Platform<TDev>, alpaka::PlatformCpu>)
      return true;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    if constexpr (std::is_same_v<TDev, alpaka::DevCudaRt>)
      return true;
#endif
    return false;
  }

  // Describe the memory of the given alpaka device, as seen by ONNX Runtime.
  template <typename TDev>
    requires(alpaka::isDevice<TDev>)::Ort::MemoryInfo
  getMemoryInfo(const TDev& device) {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    if constexpr (std::is_same_v<TDev, alpaka::DevCudaRt>)
      return ::Ort::MemoryInfo("Cuda", OrtDeviceAllocator, device.getNativeHandle(), OrtMemTypeDefault);
#endif
    // CPU, or CPU fallback for the other backends
    return ::Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  }

}  // namespace cms::Ort::alpakatools

#endif  // PhysicsTools_ONNXRuntimeAlpaka_interface_GetMemoryInfo_h
