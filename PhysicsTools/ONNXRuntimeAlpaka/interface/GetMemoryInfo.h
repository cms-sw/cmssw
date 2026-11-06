#ifndef PhysicsTools_ONNXRuntimeAlpaka_interface_GetMemoryInfo_h
#define PhysicsTools_ONNXRuntimeAlpaka_interface_GetMemoryInfo_h

#include <cstdint>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include <onnxruntime/onnxruntime_cxx_api.h>

namespace cms::Ort::alpakatools {

  // PCI vendor id of AMD, used by ONNX Runtime to identify the memory of the MIGraphX execution provider
  inline constexpr uint32_t kAMDVendorId = 0x1002;

  // Describe the memory of the given alpaka device, as seen by ONNX Runtime: the CPU, NVIDIA GPUs with the CUDA
  // execution provider, or AMD GPUs with the MIGraphX execution provider. ONNX Runtime uses it directly.
  //
  // The memory must match the device used by the execution provider, otherwise ONNX Runtime does not fail, but
  // silently copies the tensors to and from the memory of the execution provider.
  template <typename TDev>
    requires(alpaka::isDevice<TDev>)::Ort::MemoryInfo
  getMemoryInfo(const TDev& device) {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    if constexpr (std::is_same_v<TDev, alpaka::DevCudaRt>)
      return ::Ort::MemoryInfo("Cuda", OrtDeviceAllocator, device.getNativeHandle(), OrtMemTypeDefault);
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    // The MIGraphX execution provider names its device allocator "Cuda", but its device has the AMD vendor id: this
    // can only be described with the CreateMemoryInfo_V2 constructor.
    if constexpr (std::is_same_v<TDev, alpaka::DevHipRt>)
      return ::Ort::MemoryInfo("Cuda",
                               OrtMemoryInfoDeviceType_GPU,
                               kAMDVendorId,
                               static_cast<uint32_t>(device.getNativeHandle()),
                               OrtDeviceMemoryType_DEFAULT,
                               /* alignment */ 0,
                               OrtDeviceAllocator);
#endif
    // CPU
    return ::Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  }

}  // namespace cms::Ort::alpakatools

#endif  // PhysicsTools_ONNXRuntimeAlpaka_interface_GetMemoryInfo_h
