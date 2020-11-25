#ifndef HeterogeneousCore_CUDAUtilities_interface_copyAsync_h
#define HeterogeneousCore_CUDAUtilities_interface_copyAsync_h

#include <type_traits>
#include <vector>

#include "FWCore/Utilities/interface/propagate_const_array.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_noncached_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"

namespace cms {
  namespace cuda {

    // Single element

    template <typename T>
    inline void copyAsync(device::unique_ptr<T>& dst, const host::unique_ptr<T>& src, cudaStream_t stream) {
      // Shouldn't compile for array types because of sizeof(T), but
      // let's add an assert with a more helpful message
      static_assert(std::is_array<T>::value == false,
                    "For array types, use the other overload with the size parameter");
      cudaCheck(cudaMemcpyAsync(dst.get(), src.get(), sizeof(T), cudaMemcpyHostToDevice, stream));
    }

    template <typename T>
    inline void copyAsync(device::unique_ptr<T>& dst, const host::noncached::unique_ptr<T>& src, cudaStream_t stream) {
      // Shouldn't compile for array types because of sizeof(T), but
      // let's add an assert with a more helpful message
      static_assert(std::is_array<T>::value == false,
                    "For array types, use the other overload with the size parameter");
      cudaCheck(cudaMemcpyAsync(dst.get(), src.get(), sizeof(T), cudaMemcpyHostToDevice, stream));
    }

    template <typename T>
    inline void copyAsync(host::unique_ptr<T>& dst, const device::unique_ptr<T>& src, cudaStream_t stream) {
      static_assert(std::is_array<T>::value == false,
                    "For array types, use the other overload with the size parameter");
      cudaCheck(cudaMemcpyAsync(dst.get(), src.get(), sizeof(T), cudaMemcpyDeviceToHost, stream));
    }

    // Multiple elements

    template <typename T>
    inline void copyAsync(device::unique_ptr<T[]>& dst,
                          const host::unique_ptr<T[]>& src,
                          size_t nelements,
                          cudaStream_t stream) {
      cudaCheck(cudaMemcpyAsync(dst.get(), src.get(), nelements * sizeof(T), cudaMemcpyHostToDevice, stream));
    }

    template <typename T>
    inline void copyAsync(device::unique_ptr<T[]>& dst,
                          const host::noncached::unique_ptr<T[]>& src,
                          size_t nelements,
                          cudaStream_t stream) {
      cudaCheck(cudaMemcpyAsync(dst.get(), src.get(), nelements * sizeof(T), cudaMemcpyHostToDevice, stream));
    }

    template <typename T>
    inline void copyAsync(host::unique_ptr<T[]>& dst,
                          const device::unique_ptr<T[]>& src,
                          size_t nelements,
                          cudaStream_t stream) {
      cudaCheck(cudaMemcpyAsync(dst.get(), src.get(), nelements * sizeof(T), cudaMemcpyDeviceToHost, stream));
    }

    // copy from a host vector using pinned memory
    template <typename T>
    inline void copyAsync(cms::cuda::device::unique_ptr<T[]>& dst,
                          const std::vector<T, cms::cuda::HostAllocator<T>>& src,
                          cudaStream_t stream) {
      cudaCheck(cudaMemcpyAsync(dst.get(), src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
    }

    // special case used to transfer conditions data
    template <typename T>
    inline void copyAsync(edm::propagate_const_array<cms::cuda::device::unique_ptr<T[]>>& dst,
                          const std::vector<T, cms::cuda::HostAllocator<T>>& src,
                          cudaStream_t stream) {
      cudaCheck(cudaMemcpyAsync(
          get_underlying(dst).get(), src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
    }
  }  // namespace cuda
}  // namespace cms

#endif  // HeterogeneousCore_CUDAUtilities_interface_copyAsync_h
