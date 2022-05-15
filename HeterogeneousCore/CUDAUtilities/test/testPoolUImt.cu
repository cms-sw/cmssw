#include "HeterogeneousCore/CUDAUtilities/interface/cudaMemoryPool.h"

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

__global__ void kernel_set(int s, Node **p, int me) {
  int first = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = first; i < s; i += gridDim.x * blockDim.x) {
    assert(p[i]);
    auto n = p[i];
    n->it = me;
    n->i = i;
    n->p = p[i];
    n->c = 1;
  }
}

__global__ void kernel_test(int s, Node **p, int me) {
  int first = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = first; i < s; i += gridDim.x * blockDim.x) {
    assert(p[i]);
    auto n = p[i];
    atomicSub(&(n->c), 1);
    assert(n->it == me);
    assert(n->i == i);
    assert(n->p == p[i]);
    assert(0 == n->c);
  }
}
#endif

template <memoryPool::Where where>
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

  bool stop = false;
  bool bin24 = false;
  Thread monitor([&] {
    int n = 10;
    while (n--) {
      sleep(5);
      memoryPool::cuda::dumpStat();
      if (5 == n)
        bin24 = true;
    }
    std::cout << "\nstop\n" << std::endl;
    stop = true;
  });

  int s = 40;
  {
    std::cout << "try to allocate " << s << std::endl;
    auto stream = streams[0];
    {
      auto pd = memoryPool::cuda::makeBuffer<int>(s, stream, where);
      assert(pd.get());
      memoryPool::cuda::dumpStat();
      pd = memoryPool::cuda::makeBuffer<int>(s, stream, where);
      memoryPool::cuda::dumpStat();
    }
    cudaStreamSynchronize(stream);
    memoryPool::cuda::dumpStat();
  }
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
    Node **dp = nullptr;
    Node **hp = nullptr;
    cudaMalloc(&dp, 100 * sizeof(void *));
    assert(dp);
    cudaMallocHost(&hp, 100 * sizeof(void *));
    assert(hp);
#endif
    auto &stream = streams[me];

    int iter = 0;
    while (true) {
      if (stop)
        break;
      iter++;

      memoryPool::Deleter devDeleter(std::make_shared<memoryPool::cuda::BundleDelete>(stream, where));
      auto n = rgen1(eng);
      bool large = 0 == (iter % (128 + me));
      for (int k = 0; k < n; ++k) {
        int b = bin24 ? rgen24(eng) : rgen20(eng);
        // once in while let's allocate 2GB
        if (large) {
          b = 31;
          large = false;
        }
        uint64_t s = 1LL << b;
        assert(s > 0);
        try {
          auto p0 = memoryPool::cuda::makeBuffer<Node>(s / sizeof(Node) + sizeof(Node), devDeleter);
          auto p = p0.get();
          if (nullptr == p) {
            std::cout << "error not detected??? " << b << ' ' << std::endl;
            memoryPool::cuda::dumpStat();
          }
          assert(p);
          hp[k] = p;
        } catch (...) {
          std::cout << "\n\n!!!Failed " << me << " at " << iter << std::endl;
          memoryPool::cuda::dumpStat();
          return;
        }
      }
#ifdef __CUDACC__
      assert(n <= 100);
      // do something???
      cudaMemcpyAsync(dp, hp, n * sizeof(void *), cudaMemcpyHostToDevice, stream);
      kernel_set<<<1, 128, 0, stream>>>(n, dp, me);
      kernel_test<<<1, 128, 0, stream>>>(n, dp, me);

      // better sync each "event"
      // cudaStreamSynchronize(stream);
#else
      // do something???
      for (int k = 0; k < n; ++k) {
        auto p = hp[k];
        assert(p);
        auto n = p;
        n->it = me;
        n->i = i;
        n->p = p;
        n->c = 1;
      }
      for (int k = 0; k < n; ++k) {
        auto p = hp[k];
        assert(p);
        auto n = p;
        n->c--;
        assert(n->it == me);
        assert(n->i == i);
        assert(n->p == p);
        assert(0 == n->c);
      }
#endif
    }
    cudaStreamSynchronize(stream);
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
  memoryPool::cuda::dumpStat();
}

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

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
  go<memoryPool::onDevice>();
#else
  std::cout << "testing posix" << std::endl;
  go<memoryPool::onCPU>();
#endif

  return 0;
}
