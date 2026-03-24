#ifndef PhysicsTools_PyTorch_interface_PyTorchFPXBridge_h
#define PhysicsTools_PyTorch_interface_PyTorchFPXBridge_h

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)

#include <cuda_fp16.h>
#include <c10/core/ScalarType.h>

namespace c10 {

  /*
 * Map CUDA half precision type to PyTorch scalar type.
 */
  template <>
  struct CppTypeToScalarType<__half> {
    static constexpr ScalarType value = ScalarType::Half;
  };

}  // namespace c10

#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

#endif  // PhysicsTools_PyTorch_interface_PyTorchFPXBridge_h
