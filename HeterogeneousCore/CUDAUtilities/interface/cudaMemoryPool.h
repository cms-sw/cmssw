#pragma once
#include "memoryPool.h"
#include <vector>

// only for cudaStream_t
#include <cuda_runtime.h>

#include <cassert>

namespace memoryPool {
  namespace cuda {

    void dumpStat();

    SimplePoolAllocator *getPool(Where where);

    // allocate either on current device or on host
    /* inline */ std::pair<void *, int> alloc(uint64_t size, SimplePoolAllocator &pool);

    // schedule free
    /* inline */ void free(cudaStream_t stream, std::vector<int> buckets, SimplePoolAllocator &pool);

    template <typename T>
    auto copy(Buffer<T> &dst, Buffer<T> const &src, uint64_t size, cudaStream_t stream) {
      assert(dst.get());
      assert(src.get());
      assert(size > 0);
      return cudaMemcpyAsync(dst.get(), src.get(), sizeof(T) * size, cudaMemcpyDefault, stream);
    }

    struct CudaDeleterBase : public DeleterBase {
      CudaDeleterBase(cudaStream_t const &stream, Where where) : DeleterBase(getPool(where)), m_stream(stream) {
        //         if (stream) return;
        //         std::cout << "0 stream???" << std::endl;
        //         throw std::bad_alloc();
      }

      CudaDeleterBase(cudaStream_t const &stream, SimplePoolAllocator *pool) : DeleterBase(pool), m_stream(stream) {
        //           if (stream) return;
        //            std::cout << "0 stream???" << std::endl;
        //            throw std::bad_alloc();
      }

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
    Buffer<T> makeBuffer(uint64_t size, Deleter del) {
      auto ret = alloc(sizeof(T) * size, *del.pool());
      if (ret.second < 0)
        throw std::bad_alloc();
      return Buffer<T>((T *)(ret.first), ret.second, del);
    }

    template <typename T>
    Buffer<T> makeBuffer(uint64_t size, cudaStream_t const &stream, Where where) {
      return makeBuffer<T>(sizeof(T) * size, Deleter(std::make_shared<DeleteOne>(stream, getPool(where))));
    }

  }  // namespace cuda
}  // namespace memoryPool

// #include "cudaMemoryPoolImpl.h"
