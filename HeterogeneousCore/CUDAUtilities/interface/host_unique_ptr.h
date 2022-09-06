#ifndef HeterogeneousCore_CUDAUtilities_interface_host_unique_ptr_h
#define HeterogeneousCore_CUDAUtilities_interface_host_unique_ptr_h

#include <cstdlib>
#include <memory>
#include <functional>

#include "HeterogeneousCore/CUDAUtilities/interface/allocate_host.h"

namespace cms {
  namespace cuda {
    namespace host {
      namespace impl {

        enum class MemoryType : bool {
          kDefault = false,
          kPinned = true,
        };

        // Custom deleter for host memory, with an internal state to distinguish pageable and pinned host memory
        class HostDeleter {
        public:
          // The default constructor is needed by the default constructor of unique_ptr<T, HostDeleter>,
          // which is needed by the default constructor of HostProduct<T>, which is needed by the ROOT dictionary
          HostDeleter() : type_{MemoryType::kDefault} {}
          HostDeleter(MemoryType type) : type_{type} {}

          void operator()(void *ptr) {
            if (type_ == MemoryType::kPinned) {
              cms::cuda::free_host(ptr);
            } else {
              std::free(ptr);
            }
          }

        private:
          MemoryType type_;
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

    // Allocate pageable host memory
    template <typename T>
    typename host::impl::make_host_unique_selector<T>::non_array make_host_unique() {
      static_assert(std::is_trivially_constructible<T>::value,
                    "Allocating with non-trivial constructor on the host memory is not supported");
      // Allocate a buffer aligned to 128 bytes, to match the CUDA cache line size
      const size_t alignment = 128;
      // std::aligned_alloc() requires the size to be a multiple of the alignment
      const size_t size = (sizeof(T) + alignment - 1) / alignment * alignment;
      void *mem = std::aligned_alloc(alignment, size);
      return typename host::impl::make_host_unique_selector<T>::non_array{reinterpret_cast<T *>(mem),
                                                                          host::impl::MemoryType::kDefault};
    }

    template <typename T>
    typename host::impl::make_host_unique_selector<T>::unbounded_array make_host_unique(size_t n) {
      using element_type = typename std::remove_extent<T>::type;
      static_assert(std::is_trivially_constructible<element_type>::value,
                    "Allocating with non-trivial constructor on the host memory is not supported");
      // Allocate a buffer aligned to 128 bytes, to match the CUDA cache line size
      const size_t alignment = 128;
      // std::aligned_alloc() requires the size to be a multiple of the alignment
      const size_t size = (n * sizeof(element_type) + alignment - 1) / alignment * alignment;
      void *mem = std::aligned_alloc(alignment, size);
      return typename host::impl::make_host_unique_selector<T>::unbounded_array{reinterpret_cast<element_type *>(mem),
                                                                                host::impl::MemoryType::kDefault};
    }

    // Allocate pinned host memory
    template <typename T>
    typename host::impl::make_host_unique_selector<T>::non_array make_host_unique(cudaStream_t stream) {
      static_assert(std::is_trivially_constructible<T>::value,
                    "Allocating with non-trivial constructor on the host memory is not supported");
      void *mem = allocate_host(sizeof(T), stream);
      return typename host::impl::make_host_unique_selector<T>::non_array{reinterpret_cast<T *>(mem),  //
                                                                          host::impl::MemoryType::kPinned};
    }

    template <typename T>
    typename host::impl::make_host_unique_selector<T>::unbounded_array make_host_unique(size_t n, cudaStream_t stream) {
      using element_type = typename std::remove_extent<T>::type;
      static_assert(std::is_trivially_constructible<element_type>::value,
                    "Allocating with non-trivial constructor on the host memory is not supported");
      void *mem = allocate_host(n * sizeof(element_type), stream);
      return typename host::impl::make_host_unique_selector<T>::unbounded_array{reinterpret_cast<element_type *>(mem),
                                                                                host::impl::MemoryType::kPinned};
    }

    // Arrays of known bounds are not supported by std::unique_ptr
    template <typename T, typename... Args>
    typename host::impl::make_host_unique_selector<T>::bounded_array make_host_unique(Args &&...) = delete;

    // No check for the trivial constructor, make it clear in the interface
    template <typename T>
    typename host::impl::make_host_unique_selector<T>::non_array make_host_unique_uninitialized(cudaStream_t stream) {
      void *mem = allocate_host(sizeof(T), stream);
      return typename host::impl::make_host_unique_selector<T>::non_array{reinterpret_cast<T *>(mem),  //
                                                                          host::impl::MemoryType::kPinned};
    }

    template <typename T>
    typename host::impl::make_host_unique_selector<T>::unbounded_array make_host_unique_uninitialized(
        size_t n, cudaStream_t stream) {
      using element_type = typename std::remove_extent<T>::type;
      void *mem = allocate_host(n * sizeof(element_type), stream);
      return typename host::impl::make_host_unique_selector<T>::unbounded_array{reinterpret_cast<element_type *>(mem),
                                                                                host::impl::MemoryType::kPinned};
    }

    // Arrays of known bounds are not supported by std::unique_ptr
    template <typename T, typename... Args>
    typename host::impl::make_host_unique_selector<T>::bounded_array make_host_unique_uninitialized(Args &&...) = delete;

  }  // namespace cuda
}  // namespace cms

#endif
