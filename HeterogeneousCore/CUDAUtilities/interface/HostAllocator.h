#ifndef HeterogeneousCore_CUDAUtilities_HostAllocator_h
#define HeterogeneousCore_CUDAUtilities_HostAllocator_h

#include <memory>
#include <new>
#include <cuda_runtime.h>

#include "FWCore/Utilities/interface/thread_safety_macros.h"

namespace cms {
  namespace cuda {

    class bad_alloc : public std::bad_alloc {
    public:
      bad_alloc(cudaError_t error) noexcept : error_(error) {}

      const char* what() const noexcept override { return cudaGetErrorString(error_); }

    private:
      cudaError_t error_;
    };

    template <typename T, unsigned int FLAGS = cudaHostAllocDefault>
    class HostAllocator {
    public:
      using value_type = T;

      template <typename U>
      struct rebind {
        using other = HostAllocator<U, FLAGS>;
      };

      CMS_THREAD_SAFE T* allocate(std::size_t n) const __attribute__((warn_unused_result)) __attribute__((malloc))
      __attribute__((returns_nonnull)) {
        void* ptr = nullptr;
        cudaError_t status = cudaMallocHost(&ptr, n * sizeof(T), FLAGS);
        if (status != cudaSuccess) {
          throw bad_alloc(status);
        }
        if (ptr == nullptr) {
          throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
      }

      void deallocate(T* p, std::size_t n) const {
        cudaError_t status = cudaFreeHost(p);
        if (status != cudaSuccess) {
          throw bad_alloc(status);
        }
      }
    };

  }  // namespace cuda
}  // namespace cms

#endif  // HeterogeneousCore_CUDAUtilities_HostAllocator_h
