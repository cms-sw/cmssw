#pragma once
#include "memoryPool.h"
#include <vector>

// only for cudaStream_t
#include <cuda.h>

namespace memoryPool {
  namespace cuda {

    void dumpStat();

    SimplePoolAllocator *getPool(Where where);

    // allocate either on current device or on host
    std::pair<void *, int> alloc(uint64_t size, SimplePoolAllocator &pool);

    // schedule free
    void free(cudaStream_t stream, std::vector<int> buckets, SimplePoolAllocator &pool);

    template<typename T>
    auto copy(buffer<T> & dst,buffer<T> const & src, uint64_t size, cudaStream_t stream) {
        return cudaMemcpyAsync(dst.get(), src.get(), sizeof(T)*size,  cudaMemcpyDefault, stream);
    }

    struct CudaDeleterBase : public DeleterBase {
      CudaDeleterBase(cudaStream_t const &stream, Where where) : DeleterBase(getPool(where)), m_stream(stream) {}

      CudaDeleterBase(cudaStream_t const &stream, SimplePoolAllocator *pool) : DeleterBase(pool), m_stream(stream) {}

      ~CudaDeleterBase() override = default;

      cudaStream_t m_stream;
    };

    struct DeleteOne final : public CudaDeleterBase {
      using CudaDeleterBase::CudaDeleterBase;

      ~DeleteOne() override = default;
      void operator()(int bucket) override { free(m_stream, std::vector<int>(1, bucket), *pool()); }
    };

    struct BundleDelete final : public CudaDeleterBase {
      using CudaDeleterBase::CudaDeleterBase;

      ~BundleDelete() override { free(m_stream, std::move(m_buckets), *pool()); }

      void operator()(int bucket) override { m_buckets.push_back(bucket); }

      std::vector<int> m_buckets;
    };

    template <typename T>
    buffer<T> make_buffer(uint64_t size, Deleter del) {
      auto ret = alloc(sizeof(T) * size, *del.pool());
      if (ret.second < 0)
        throw std::bad_alloc();
      del.setBucket(ret.second);
      return buffer<T>((T *)(ret.first), del);
    }

    template <typename T>
    buffer<T> make_buffer(uint64_t size, cudaStream_t const &stream, Where where) {
      return make_buffer<T>(sizeof(T) * size, Deleter(std::make_shared<DeleteOne>(stream, getPool(where))));
    }

    /*
      template< class T, class... Args >
      memoryPool::buffer<T> make_buffer( Args&&... args );
      template< class T, class... Args >
      memoryPool::buffer<T> make_buffer(Deleter del, Args&&... args );
*/

  }  // namespace cuda
}  // namespace memoryPool
