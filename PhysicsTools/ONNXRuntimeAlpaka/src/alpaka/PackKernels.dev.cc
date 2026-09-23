#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "PhysicsTools/ONNXRuntimeAlpaka/interface/alpaka/PackKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ort::detail {

  using cms::Ort::alpakatools::PackDescriptor;

  // offset (in elements) in the SoA of the element at position `index` in the row-major buffer
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int64_t soaOffset(PackDescriptor const& desc, int64_t features, int64_t index) {
    int64_t sample = index / features;
    int64_t feature = index % features;
    int64_t offset = sample * desc.batch_stride;
    for (int d = desc.n_dims - 1; d >= 0; --d) {
      offset += (feature % desc.sizes[d]) * desc.strides[d];
      feature /= desc.sizes[d];
    }
    return offset;
  }

  template <typename T>
  struct PackKernel {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, PackDescriptor desc, T const* soa, T* buffer) const {
      const int64_t features = desc.features();
      for (int64_t index : cms::alpakatools::uniform_elements(acc, desc.elements())) {
        buffer[index] = soa[soaOffset(desc, features, index)];
      }
    }
  };

  template <typename T>
  struct UnpackKernel {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, PackDescriptor desc, T const* buffer, T* soa) const {
      const int64_t features = desc.features();
      for (int64_t index : cms::alpakatools::uniform_elements(acc, desc.elements())) {
        soa[soaOffset(desc, features, index)] = buffer[index];
      }
    }
  };

  // The kernels only move data around, so they are instantiated for the element sizes rather than for the types.
  template <template <typename> class TKernel>
  void launch(Queue& queue, PackDescriptor const& desc, std::byte const* from, std::byte* to) {
    if (desc.elements() == 0)
      return;
    const uint32_t threads = 256;
    const uint32_t blocks = static_cast<uint32_t>((desc.elements() + threads - 1) / threads);
    const auto workdiv = cms::alpakatools::make_workdiv<Acc1D>(blocks, threads);
    switch (desc.elem_size) {
      case 1:
        alpaka::exec<Acc1D>(queue,
                            workdiv,
                            TKernel<uint8_t>{},
                            desc,
                            reinterpret_cast<uint8_t const*>(from),
                            reinterpret_cast<uint8_t*>(to));
        break;
      case 2:
        alpaka::exec<Acc1D>(queue,
                            workdiv,
                            TKernel<uint16_t>{},
                            desc,
                            reinterpret_cast<uint16_t const*>(from),
                            reinterpret_cast<uint16_t*>(to));
        break;
      case 4:
        alpaka::exec<Acc1D>(queue,
                            workdiv,
                            TKernel<uint32_t>{},
                            desc,
                            reinterpret_cast<uint32_t const*>(from),
                            reinterpret_cast<uint32_t*>(to));
        break;
      case 8:
        alpaka::exec<Acc1D>(queue,
                            workdiv,
                            TKernel<uint64_t>{},
                            desc,
                            reinterpret_cast<uint64_t const*>(from),
                            reinterpret_cast<uint64_t*>(to));
        break;
      default:
        throw cms::Exception("UnsupportedType")
            << "Tensors with elements of " << desc.elem_size << " bytes are not supported.";
    }
  }

  void pack(Queue& queue, PackDescriptor const& desc, std::byte const* soa, std::byte* buffer) {
    launch<PackKernel>(queue, desc, soa, buffer);
  }

  void unpack(Queue& queue, PackDescriptor const& desc, std::byte const* buffer, std::byte* soa) {
    launch<UnpackKernel>(queue, desc, buffer, soa);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ort::detail
