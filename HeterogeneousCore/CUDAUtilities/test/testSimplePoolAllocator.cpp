#include "HeterogeneousCore/CUDAUtilities/interface/SimplePoolAllocator.h"

#include <cmath>
#include <unistd.h>

#include <random>
#include <limits>

#include <atomic>
#include <thread>
#include <mutex>

typedef std::thread Thread;
typedef std::vector<std::thread> ThreadGroup;
typedef std::mutex Mutex;
typedef std::lock_guard<std::mutex> Lock;

struct Node {
  int it = -1;
  int i = -1;
  void *p = nullptr;
#ifdef __CUDACC__
  int c = 0;
#else
  std::atomic<int> c = 0;
#endif
};

#ifdef __CUDACC__

// generic callback
template <typename F>
void CUDART_CB myCallback(void *fun) {
  (*(F *)(fun))();
}

__global__ void kernel_set(int s, void **p, int me) {
  int first = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = first; i < s; i += gridDim.x * blockDim.x) {
    assert(p[i]);
    auto n = (Node *)(p[i]);
    n->it = me;
    n->i = i;
    n->p = p[i];
    n->c = 1;
  }
}

__global__ void kernel_test(int s, void **p, int me) {
  int first = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = first; i < s; i += gridDim.x * blockDim.x) {
    assert(p + i);
    auto n = (Node *)(p[i]);
    atomicSub(&(n->c), 1);
    assert(n->it == me);
    assert(n->i == i);
    assert(n->p == p[i]);
    assert(0 == n->c);
  }
}
#endif

template <typename Traits>
void go() {
  auto start = std::chrono::high_resolution_clock::now();

  const int NUMTHREADS = 24;

#ifdef __CUDACC__
  printf("Using CUDA %d\n", CUDART_VERSION);
  int cuda_device = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, cuda_device);
  printf("CUDA Capable: SM %d.%d hardware\n", deviceProp.major, deviceProp.minor);

  cudaStream_t streams[NUMTHREADS];

  for (int i = 0; i < NUMTHREADS; i++) {
    cudaStreamCreate(&(streams[i]));
  }

#endif

  SimplePoolAllocatorImpl<Traits> pool(128 * 1024);
  assert(0 == pool.size());

  bool stop = false;
  bool bin24 = false;
  Thread monitor([&] {
    int n = 10;
    while (n--) {
      sleep(5);
      pool.dumpStat();
      if (5 == n)
        bin24 = true;
    }
    std::cout << "\nstop\n" << std::endl;
    stop = true;
  });

  int s = 40;

  std::cout << "try to allocate " << s << std::endl;

  int i0 = pool.alloc(s);
  assert(1 == pool.size());
  assert(i0 >= 0);
  auto p0 = pool.pointer(i0);
  assert(nullptr != p0);

  pool.free(i0);
  assert(1 == pool.size());

  int i1 = pool.alloc(s);
  assert(1 == pool.size());
  assert(i1 == i0);
  auto p1 = pool.pointer(i1);
  assert(p1 == p0);

  std::atomic<int> nt = 0;

  auto test = [&] {
    int const me = nt++;
    auto delta = std::chrono::high_resolution_clock::now() - start;

    std::mt19937 eng(me + std::chrono::duration_cast<std::chrono::milliseconds>(delta).count());
    std::uniform_int_distribution<int> rgen1(1, 100);
    std::uniform_int_distribution<int> rgen20(3, 20);
    std::uniform_int_distribution<int> rgen24(3, 24);
    std::cout << "first RN " << rgen1(eng) << " at "
              << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count() << " in " << me << std::endl;

#ifdef __CUDACC__
    void **dp = nullptr;
    void **hp = nullptr;
    cudaMalloc(&dp, 100 * sizeof(void *));
    assert(dp);
    cudaMallocHost(&hp, 100 * sizeof(void *));
    assert(hp);
#endif

    int iter = 0;
    while (true) {
      if (stop)
        break;
      iter++;
      auto n = rgen1(eng);
      int ind[n];
      bool large = 0 == (iter % (128 + me));
      for (auto &i : ind) {
        int b = bin24 ? rgen24(eng) : rgen20(eng);
        // once in while let's allocate 2GB
        if (large) {
          b = 31;
          large = false;
        }
        uint64_t s = 1LL << b;
        assert(s > 0);
        i = pool.alloc(s + sizeof(Node));
        if (i < 0) {
          std::cout << "\n\n!!!Failed " << me << " at " << iter << std::endl;
          pool.dumpStat();
          return;
        }
        assert(i >= 0);
        auto p = pool.pointer(i);
        if (nullptr == p) {
          std::cout << "error not detected??? " << b << ' ' << i << std::endl;
          pool.dumpStat();
        }
        assert(p);
      }
#ifdef __CUDACC__
      assert(n <= 100);
      auto &stream = streams[me];
      // do something???
      for (int k = 0; k < n; ++k) {
        auto i = ind[k];
        hp[k] = pool.pointer(i);
      }
      cudaMemcpyAsync(dp, hp, n * sizeof(void *), cudaMemcpyHostToDevice, stream);
      kernel_set<<<1, 128, 0, stream>>>(n, dp, me);
      kernel_test<<<1, 128, 0, stream>>>(n, dp, me);

      // free
      auto doFree = [&]() {
        for (int k = 0; k < n; ++k) {
          auto i = ind[k];
          pool.free(i);
        }
      };
      cudaLaunchHostFunc(stream, myCallback<decltype(doFree)>, &doFree);

      // better sync each "event"
      cudaStreamSynchronize(stream);
#else
      // do something???
      for (auto i : ind) {
        auto p = pool.pointer(i);
        assert(p);
        auto n = (Node *)(p);
        n->it = me;
        n->i = i;
        n->p = p;
        n->c = 1;
      }
      for (auto i : ind) {
        auto p = pool.pointer(i);
        assert(p);
        auto n = (Node *)(p);
        n->c--;
        assert(n->it == me);
        assert(n->i == i);
        assert(n->p == p);
        assert(0 == n->c);
      }
      // free
      for (auto i : ind) {
        pool.free(i);
      }
#endif
    }
  };

  ThreadGroup threads;
  threads.reserve(NUMTHREADS);

  for (int i = 0; i < NUMTHREADS; ++i) {
    threads.emplace_back(test);
  }

  for (auto &t : threads)
    t.join();

  threads.clear();
  monitor.join();
  std::cout << "\nfinished\n" << std::endl;
  pool.dumpStat();
}

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

struct CudaDeviceAlloc {
  using Pointer = void *;

  static Pointer alloc(size_t size) {
    Pointer p = nullptr;
    auto err = cudaMalloc(&p, size);
    return err == cudaSuccess ? p : nullptr;
  }
  static void free(Pointer ptr) { cudaFree(ptr); }
};

struct CudaHostAlloc {
  using Pointer = void *;

  static Pointer alloc(size_t size) {
    Pointer p = nullptr;
    auto err = cudaMallocHost(&p, size);
    return err == cudaSuccess ? p : nullptr;
  }
  static void free(Pointer ptr) { cudaFreeHost(ptr); }
};
#endif

int main() {
#ifdef __CUDACC__
  {
    int devices = 0;
    auto status = cudaGetDeviceCount(&devices);
    if (status != cudaSuccess || 0 == devices)
      return 0;
    std::cout << "found " << devices << " cuda devices" << std::endl;
  }

  std::cout << "\ntesting cuda device" << std::endl;
  go<CudaDeviceAlloc>();
#else
  std::cout << "testing posix" << std::endl;
  go<PosixAlloc>();
#endif

  return 0;
}
