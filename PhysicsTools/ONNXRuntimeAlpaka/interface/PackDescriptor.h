#ifndef PhysicsTools_ONNXRuntimeAlpaka_interface_PackDescriptor_h
#define PhysicsTools_ONNXRuntimeAlpaka_interface_PackDescriptor_h

#include <cstdint>

namespace cms::Ort::alpakatools {

  // Describes how the elements of a tensor are laid out in the SoA memory, in units of elements:
  //   element (i, m_0, ..., m_{n-1}) is at offset  i * batch_stride + sum_d m_d * strides[d]
  // A packed tensor is the same data stored as a contiguous, row-major [batch, sizes[0], ..., sizes[n-1]] array,
  // which is the only layout supported by ONNX Runtime.
  struct PackDescriptor {
    static constexpr int kMaxDims = 4;

    int32_t elem_size = 0;     // size of each element, in bytes
    int32_t batch = 0;         // number of elements along the batch dimension
    int64_t batch_stride = 1;  // 1 for columns, 0 for scalars (broadcasted along the batch dimension)
    int32_t n_dims = 0;        // number of non-batch dimensions
    int64_t sizes[kMaxDims] = {};
    int64_t strides[kMaxDims] = {};

    constexpr int64_t features() const {
      int64_t volume = 1;
      for (int d = 0; d < n_dims; ++d)
        volume *= sizes[d];
      return volume;
    }

    constexpr int64_t elements() const { return static_cast<int64_t>(batch) * features(); }
    constexpr int64_t bytes() const { return elements() * elem_size; }
  };

}  // namespace cms::Ort::alpakatools

#endif  // PhysicsTools_ONNXRuntimeAlpaka_interface_PackDescriptor_h
