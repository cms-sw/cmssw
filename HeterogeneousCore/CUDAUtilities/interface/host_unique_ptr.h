#ifndef HeterogeneousCore_CUDAUtilities_interface_host_unique_ptr_h
#define HeterogeneousCore_CUDAUtilities_interface_host_unique_ptr_h

#include <memory>
#include <functional>

#include "HeterogeneousCore/CUDAUtilities/interface/allocate_host.h"

namespace cms {
  namespace cuda {
    namespace host {
      namespace impl {
        // Additional layer of types to distinguish from host::unique_ptr
        class HostDeleter {
        public:
          void operator()(void *ptr) { cms::cuda::free_host(ptr); }
        };
      }  // namespace impl

      template <typename T>
      using unique_ptr = std::unique_ptr<T, impl::HostDeleter>;

      namespace impl {
        template <typename T>
        struct make_host_unique_selector {
          using non_array = cms::cuda::host::unique_ptr<T>;
        };
        template <typename T>
        struct make_host_unique_selector<T[]> {
          using unbounded_array = cms::cuda::host::unique_ptr<T[]>;
        };
        template <typename T, size_t N>
        struct make_host_unique_selector<T[N]> {
          struct bounded_array {};
        };
      }  // namespace impl
    }    // namespace host

    // Allocate pinned host memory
    template <typename T>
    typename host::impl::make_host_unique_selector<T>::non_array make_host_unique(cudaStream_t stream) {
      static_assert(std::is_trivially_constructible<T>::value,
                    "Allocating with non-trivial constructor on the pinned host memory is not supported");
      void *mem = allocate_host(sizeof(T), stream);
      return typename host::impl::make_host_unique_selector<T>::non_array{reinterpret_cast<T *>(mem)};
    }

    template <typename T>
    typename host::impl::make_host_unique_selector<T>::unbounded_array make_host_unique(size_t n, cudaStream_t stream) {
      using element_type = typename std::remove_extent<T>::type;
      static_assert(std::is_trivially_constructible<element_type>::value,
                    "Allocating with non-trivial constructor on the pinned host memory is not supported");
      void *mem = allocate_host(n * sizeof(element_type), stream);
      return typename host::impl::make_host_unique_selector<T>::unbounded_array{reinterpret_cast<element_type *>(mem)};
    }

    template <typename T, typename... Args>
    typename host::impl::make_host_unique_selector<T>::bounded_array make_host_unique(Args &&...) = delete;

    // No check for the trivial constructor, make it clear in the interface
    template <typename T>
    typename host::impl::make_host_unique_selector<T>::non_array make_host_unique_uninitialized(cudaStream_t stream) {
      void *mem = allocate_host(sizeof(T), stream);
      return typename host::impl::make_host_unique_selector<T>::non_array{reinterpret_cast<T *>(mem)};
    }

    template <typename T>
    typename host::impl::make_host_unique_selector<T>::unbounded_array make_host_unique_uninitialized(
        size_t n, cudaStream_t stream) {
      using element_type = typename std::remove_extent<T>::type;
      void *mem = allocate_host(n * sizeof(element_type), stream);
      return typename host::impl::make_host_unique_selector<T>::unbounded_array{reinterpret_cast<element_type *>(mem)};
    }

    template <typename T, typename... Args>
    typename host::impl::make_host_unique_selector<T>::bounded_array make_host_unique_uninitialized(Args &&...) = delete;
  }  // namespace cuda
}  // namespace cms

#endif
