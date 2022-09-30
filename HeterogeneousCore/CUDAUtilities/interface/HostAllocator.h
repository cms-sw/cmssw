#ifndef HeterogeneousCore_CUDAUtilities_HostAllocator_h
#define HeterogeneousCore_CUDAUtilities_HostAllocator_h

#include <memory>
#include <new>
#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaMemoryPool.h"
#include "HeterogeneousCore/CUDAUtilities/interface/SimplePoolAllocator.h"

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

      typedef std::true_type propagate_on_container_copy_assignment;
      typedef std::true_type propagate_on_container_move_assignment;
      typedef std::true_type propagate_on_container_swap;

      HostAllocator() = default;
      HostAllocator(cudaStream_t stream) : m_stream(stream) {}

      bool operator==(HostAllocator<T, FLAGS> const& rh) const { return (m_stream) == (rh.m_stream); }
      bool operator!=(HostAllocator<T, FLAGS> const& rh) const { return (m_stream) != (rh.m_stream); }

      CMS_THREAD_SAFE T* allocate(std::size_t n) const __attribute__((warn_unused_result)) __attribute__((malloc))
      __attribute__((returns_nonnull)) {
        assert(n > 0);
        void* ptr = nullptr;
        if (!m_stream) {
          cudaError_t status = cudaMallocHost(&ptr, n * sizeof(T), FLAGS);
          if (status != cudaSuccess) {
            throw bad_alloc(status);
          }
        } else {
          auto pool = memoryPool::cuda::getPool(memoryPool::onHost);
          assert(pool);
          int i = pool->alloc(sizeof(T) * n, stream());
          if (i >= 0)
            ptr = pool->pointer(i);
        }
        if (ptr == nullptr) {
          throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
      }

      void deallocate(T* p, std::size_t n) const {
        if (!p)
          return;
        if (!m_stream) {
          cudaError_t status = cudaFreeHost(p);
          if (status != cudaSuccess) {
            throw bad_alloc(status);
          }
        } else {
          auto pool = memoryPool::cuda::getPool(memoryPool::onHost);
          assert(pool);
          auto i = pool->index(p);
          assert(i >= 0);
          auto c = pool->count(i);
          memoryPool::cuda::free(stream(), std::vector<std::pair<int, uint64_t>>(1, std::make_pair(i, c)), *pool);
        }
      }

      void setStream(cudaStream_t stream) { m_stream = stream; }
      cudaStream_t stream() const { return m_stream; }
      cudaStream_t m_stream = nullptr;
    };

    template <typename T>
    void resizeContainer(T& c, size_t size, cudaStream_t stream) {
      c = T(size, typename T::allocator_type(stream));
    }

  }  // namespace cuda
}  // namespace cms

#endif  // HeterogeneousCore_CUDAUtilities_HostAllocator_h
