#pragma once
#include "memoryPool.h"
#include "SimplePoolAllocator.h"
#include <vector>

// only for cudaStream_t
#include <cuda_runtime.h>

#include <cassert>

namespace memoryPool {
  namespace cuda {

    void init(bool onlyCPU = false);
    void shutdown();

    void dumpStat();

    SimplePoolAllocator *getPool(Where where);

    // allocate either on current device or on host
    std::tuple<void *, int, uint64_t> alloc(void *stream, uint64_t size, SimplePoolAllocator &pool);

    // schedule free
    void free(void *stream, std::vector<std::pair<int, uint64_t>> buckets, SimplePoolAllocator &pool);

    template <typename T>
    auto copy(Buffer<T> &dst, Buffer<T> const &src, uint64_t size, cudaStream_t stream) {
      assert(dst.get());
      assert(src.get());
      assert(size > 0);
      return cudaMemcpyAsync(dst.get(), src.get(), sizeof(T) * size, cudaMemcpyDefault, stream);
    }

    struct CudaDeleterBase : public DeleterBase {
      CudaDeleterBase(void *stream, Where where) : DeleterBase(getPool(where), stream) {}

      CudaDeleterBase(void *stream, SimplePoolAllocator *pool) : DeleterBase(pool, stream) {}

      ~CudaDeleterBase() override = default;
    };

    struct DeleteOne final : public CudaDeleterBase {
      using CudaDeleterBase::CudaDeleterBase;

      ~DeleteOne() override = default;
      void operator()(int bucket, uint64_t count) override {
        free(stream(), std::vector<std::pair<int, uint64_t>>(1, std::make_pair(bucket, count)), *pool());
      }
    };

    // in DataFormat  is safe to  free immediately as stream sync already happened (what about "strange" early delete???)
    struct ImmediateDelete final : public CudaDeleterBase {
      using CudaDeleterBase::CudaDeleterBase;

      ~ImmediateDelete() override = default;
      void operator()(int bucket, uint64_t count) override {
        pool()->setScheduled(bucket);
        pool()->free(bucket, count);
      }
    };

    struct BundleDelete final : public CudaDeleterBase {
      BundleDelete(void *stream, Where where) : CudaDeleterBase(stream, where) { m_buckets.reserve(8); }

      ~BundleDelete() override { free(stream(), std::move(m_buckets), *pool()); }

      void operator()(int bucket, uint64_t count) override { m_buckets.push_back(std::make_pair(bucket, count)); }

      std::vector<std::pair<int, uint64_t>> m_buckets;
    };

    template <typename T>
    Buffer<T> makeBuffer(uint64_t size, Deleter const &del) {
      auto ret = alloc(del.stream(), sizeof(T) * size, *del.pool());
      if (std::get<1>(ret) < 0) {
        std::cout << "could not allocate " << size << ' ' << typeid(T).name() << " of size " << sizeof(T) << std::endl;
        throw std::bad_alloc();
      }
      return Buffer<T>((T *)(std::get<0>(ret)), std::get<1>(ret), std::get<2>(ret), del);
    }

    template <typename T>
    Buffer<T> makeBuffer(uint64_t size, Deleter &&del) {
      auto ret = alloc(del.stream(), sizeof(T) * size, *del.pool());
      if (std::get<1>(ret) < 0) {
        std::cout << "could not allocate " << size << ' ' << typeid(T).name() << " of size " << sizeof(T) << std::endl;
        throw std::bad_alloc();
      }
      return Buffer<T>((T *)(std::get<0>(ret)), std::get<1>(ret), std::get<2>(ret), std::move(del));
    }

    template <typename T>
    Buffer<T> makeBuffer(uint64_t size, void *stream, Where where) {
      return makeBuffer<T>(size, Deleter(std::make_shared<DeleteOne>(stream, where)));
    }

  }  // namespace cuda
}  // namespace memoryPool
