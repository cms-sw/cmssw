#include <cuda/api_wrappers.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "TestHeterogeneousEDProducerGPUHelpers.h"

//
// Vector Addition Kernel
//
namespace {
  template<typename T>
  __global__
  void vectorAdd(const T *a, const T *b, T *c, int numElements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < numElements) { c[i] = a[i] + b[i]; }
  }

  template <typename T>
  __global__
  void vectorProd(const T *a, const T *b, T *c, int numElements) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if(row < numElements && col < numElements) {
      c[row*numElements + col] = a[row]*b[col];
    }
  }

  template <typename T>
  __global__
  void matrixMul(const T *a, const T *b, T *c, int numElements) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if(row < numElements && col < numElements) {
      T tmp = 0;
      for(int i=0; i<numElements; ++i) {
        tmp += a[row*numElements + i] * b[i*numElements + col];
      }
      c[row*numElements + col] = tmp;
    }
  }

  template <typename T>
  __global__
  void matrixMulVector(const T *a, const T *b, T *c, int numElements) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    if(row < numElements) {
      T tmp = 0;
      for(int i=0; i<numElements; ++i) {
        tmp += a[row*numElements + i] * b[i];
      }
      c[row] = tmp;
    }
  }
}

int TestAcceleratorServiceProducerGPUHelpers_simple_kernel(int input) {
  // Example from Viktor/cuda-api-wrappers
  constexpr int NUM_VALUES = 10000;

  auto current_device = cuda::device::current::get();
  auto stream = current_device.create_stream(cuda::stream::implicitly_synchronizes_with_default_stream);

  auto h_a = cuda::memory::host::make_unique<int[]>(NUM_VALUES);
  auto h_b = cuda::memory::host::make_unique<int[]>(NUM_VALUES);
  auto h_c = cuda::memory::host::make_unique<int[]>(NUM_VALUES);

  for (auto i=0; i<NUM_VALUES; i++) {
    h_a[i] = input + i;
    h_b[i] = i*i;
  }

  auto d_a = cuda::memory::device::make_unique<int[]>(current_device, NUM_VALUES);
  auto d_b = cuda::memory::device::make_unique<int[]>(current_device, NUM_VALUES);
  auto d_c = cuda::memory::device::make_unique<int[]>(current_device, NUM_VALUES);

  cuda::memory::async::copy(d_a.get(), h_a.get(), NUM_VALUES*sizeof(int), stream.id());
  cuda::memory::async::copy(d_b.get(), h_b.get(), NUM_VALUES*sizeof(int), stream.id());

  int threadsPerBlock {256};
  int blocksPerGrid = (NUM_VALUES + threadsPerBlock - 1) / threadsPerBlock;

  vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream.id()>>>(d_a.get(), d_b.get(), d_c.get(), NUM_VALUES);
  cudaCheck(cudaGetLastError());
  /*
    // doesn't work with header-only?
  cuda::launch(vectorAdd, {blocksPerGrid, threadsPerBlock},
               d_a.get(), d_b.get(), d_c.get(), NUM_VALUES);
  */

  cuda::memory::async::copy(h_c.get(), d_c.get(), NUM_VALUES*sizeof(int), stream.id());

  stream.synchronize();

  int ret = 0;
  for (auto i=0; i<10; i++) {
    ret += h_c[i];
  }

  return ret;
}

namespace {
  constexpr int NUM_VALUES = 10000;
}

TestHeterogeneousEDProducerGPUTask::TestHeterogeneousEDProducerGPUTask() {
  h_a = cuda::memory::host::make_unique<float[]>(NUM_VALUES);
  h_b = cuda::memory::host::make_unique<float[]>(NUM_VALUES);

  auto current_device = cuda::device::current::get();
  d_b = cuda::memory::device::make_unique<float[]>(current_device, NUM_VALUES);

  d_ma = cuda::memory::device::make_unique<float[]>(current_device, NUM_VALUES*NUM_VALUES);
  d_mb = cuda::memory::device::make_unique<float[]>(current_device, NUM_VALUES*NUM_VALUES);
  d_mc = cuda::memory::device::make_unique<float[]>(current_device, NUM_VALUES*NUM_VALUES);
}

TestHeterogeneousEDProducerGPUTask::ResultType
TestHeterogeneousEDProducerGPUTask::runAlgo(const std::string& label, int input, const ResultTypeRaw inputArrays, cuda::stream_t<>& stream) {
  // First make the sanity check
  if(inputArrays.first != nullptr) {
    auto h_check = std::make_unique<float[]>(NUM_VALUES);
    cuda::memory::copy(h_check.get(), inputArrays.first, NUM_VALUES*sizeof(float));
    for(int i=0; i<NUM_VALUES; ++i) {
      if(h_check[i] != i) {
        throw cms::Exception("Assert") << "Sanity check on element " << i << " failed, expected " << i << " got " << h_check[i];
      }
    }
  }

  for (auto i=0; i<NUM_VALUES; i++) {
    h_a[i] = i;
    h_b[i] = i*i;
  }

  auto current_device = cuda::device::current::get();
  auto d_a = cuda::memory::device::make_unique<float[]>(current_device, NUM_VALUES);
  auto d_c = cuda::memory::device::make_unique<float[]>(current_device, NUM_VALUES);
  if(inputArrays.second != nullptr) {
    d_d = cuda::memory::device::make_unique<float[]>(current_device, NUM_VALUES);
  }

  // Create stream
  cuda::memory::async::copy(d_a.get(), h_a.get(), NUM_VALUES*sizeof(float), stream.id());
  cuda::memory::async::copy(d_b.get(), h_b.get(), NUM_VALUES*sizeof(float), stream.id());

  int threadsPerBlock {32};
  int blocksPerGrid = (NUM_VALUES + threadsPerBlock - 1) / threadsPerBlock;

  edm::LogPrint("TestHeterogeneousEDProducerGPU") << "  " << label << " GPU launching kernels device " << current_device.id() << " CUDA stream " << stream.id();
  vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream.id()>>>(d_a.get(), d_b.get(), d_c.get(), NUM_VALUES);
  cudaCheck(cudaGetLastError());
  if(inputArrays.second != nullptr) {
    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream.id()>>>(inputArrays.second, d_c.get(), d_d.get(), NUM_VALUES);
    cudaCheck(cudaGetLastError());
    std::swap(d_c, d_d);
  }

  dim3 threadsPerBlock3{NUM_VALUES, NUM_VALUES};
  dim3 blocksPerGrid3{1,1};
  if(NUM_VALUES*NUM_VALUES > 32) {
    threadsPerBlock3.x = 32;
    threadsPerBlock3.y = 32;
    blocksPerGrid3.x = ceil(double(NUM_VALUES)/double(threadsPerBlock3.x));
    blocksPerGrid3.y = ceil(double(NUM_VALUES)/double(threadsPerBlock3.y));
  }
  vectorProd<<<blocksPerGrid3, threadsPerBlock3, 0, stream.id()>>>(d_a.get(), d_b.get(), d_ma.get(), NUM_VALUES);
  cudaCheck(cudaGetLastError());
  vectorProd<<<blocksPerGrid3, threadsPerBlock3, 0, stream.id()>>>(d_a.get(), d_c.get(), d_mb.get(), NUM_VALUES);
  cudaCheck(cudaGetLastError());
  matrixMul<<<blocksPerGrid3, threadsPerBlock3, 0, stream.id()>>>(d_ma.get(), d_mb.get(), d_mc.get(), NUM_VALUES);
  cudaCheck(cudaGetLastError());
  matrixMulVector<<<blocksPerGrid, threadsPerBlock, 0, stream.id()>>>(d_mc.get(), d_b.get(), d_c.get(), NUM_VALUES);
  cudaCheck(cudaGetLastError());

  edm::LogPrint("TestHeterogeneousEDProducerGPU") << "  " << label << " GPU kernels launched, returning return pointer device " << current_device.id() << " CUDA stream " << stream.id();
  return std::make_pair(std::move(d_a), std::move(d_c));
}

void TestHeterogeneousEDProducerGPUTask::release(const std::string& label, cuda::stream_t<>& stream) {
  // any way to automate the release?
  edm::LogPrint("TestHeterogeneousEDProducerGPU") << "  " << label << " GPU releasing temporary memory device " << cuda::stream::associated_device(stream.id()) << " CUDA stream " << stream.id();
  d_d.reset();
}

int TestHeterogeneousEDProducerGPUTask::getResult(const ResultTypeRaw& d_ac, cuda::stream_t<>& stream) {
  auto h_c = cuda::memory::host::make_unique<float[]>(NUM_VALUES);
  cuda::memory::async::copy(h_c.get(), d_ac.second, NUM_VALUES*sizeof(int), stream.id());
  stream.synchronize();

  float ret = 0;
  for (auto i=0; i<NUM_VALUES; i++) {
    ret += h_c[i];
  }

  return static_cast<int>(ret);
}

