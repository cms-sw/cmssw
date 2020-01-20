#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/CUDAUtilities/interface/MessageLogger.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/currentDevice.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

#include "TestCUDAProducerGPUKernel.h"

namespace {
  template <typename T>
  __global__ void vectorAddConstant(T *a, T b, int numElements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numElements) {
      a[i] += b;
    }
  }

  template <typename T>
  __global__ void vectorAdd(const T *a, const T *b, T *c, int numElements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numElements) {
      c[i] = a[i] + b[i];
    }
  }

  template <typename T>
  __global__ void vectorProd(const T *a, const T *b, T *c, int numElements) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numElements && col < numElements) {
      c[row * numElements + col] = a[row] * b[col];
    }
  }

  template <typename T>
  __global__ void matrixMul(const T *a, const T *b, T *c, int numElements) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numElements && col < numElements) {
      T tmp = 0;
      for (int i = 0; i < numElements; ++i) {
        tmp += a[row * numElements + i] * b[i * numElements + col];
      }
      c[row * numElements + col] = tmp;
    }
  }

  template <typename T>
  __global__ void matrixMulVector(const T *a, const T *b, T *c, int numElements) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < numElements) {
      T tmp = 0;
      for (int i = 0; i < numElements; ++i) {
        tmp += a[row * numElements + i] * b[i];
      }
      c[row] = tmp;
    }
  }
}  // namespace

cms::cuda::device::unique_ptr<float[]> TestCUDAProducerGPUKernel::runAlgo(const std::string &label,
                                                                          const float *d_input,
                                                                          cudaStream_t stream) const {
  // First make the sanity check
  if (d_input != nullptr) {
    auto h_check = std::make_unique<float[]>(NUM_VALUES);
    cudaCheck(cudaMemcpyAsync(h_check.get(), d_input, NUM_VALUES * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    for (int i = 0; i < NUM_VALUES; ++i) {
      if (h_check[i] != i) {
        throw cms::Exception("Assert") << "Sanity check on element " << i << " failed, expected " << i << " got "
                                       << h_check[i];
      }
    }
  }

  auto h_a = cms::cuda::make_host_unique<float[]>(NUM_VALUES, stream);
  auto h_b = cms::cuda::make_host_unique<float[]>(NUM_VALUES, stream);

  for (auto i = 0; i < NUM_VALUES; i++) {
    h_a[i] = i;
    h_b[i] = i * i;
  }

  auto d_a = cms::cuda::make_device_unique<float[]>(NUM_VALUES, stream);
  auto d_b = cms::cuda::make_device_unique<float[]>(NUM_VALUES, stream);

  cudaCheck(cudaMemcpyAsync(d_a.get(), h_a.get(), NUM_VALUES * sizeof(float), cudaMemcpyHostToDevice, stream));
  cudaCheck(cudaMemcpyAsync(d_b.get(), h_b.get(), NUM_VALUES * sizeof(float), cudaMemcpyHostToDevice, stream));

  int threadsPerBlock{32};
  int blocksPerGrid = (NUM_VALUES + threadsPerBlock - 1) / threadsPerBlock;

  auto d_c = cms::cuda::make_device_unique<float[]>(NUM_VALUES, stream);
  auto current_device = cms::cuda::currentDevice();
  cms::cuda::LogVerbatim("TestHeterogeneousEDProducerGPU")
      << "  " << label << " GPU launching kernels device " << current_device << " CUDA stream " << stream;
  vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_a.get(), d_b.get(), d_c.get(), NUM_VALUES);

  auto d_ma = cms::cuda::make_device_unique<float[]>(NUM_VALUES * NUM_VALUES, stream);
  auto d_mb = cms::cuda::make_device_unique<float[]>(NUM_VALUES * NUM_VALUES, stream);
  auto d_mc = cms::cuda::make_device_unique<float[]>(NUM_VALUES * NUM_VALUES, stream);
  dim3 threadsPerBlock3{NUM_VALUES, NUM_VALUES};
  dim3 blocksPerGrid3{1, 1};
  if (NUM_VALUES * NUM_VALUES > 32) {
    threadsPerBlock3.x = 32;
    threadsPerBlock3.y = 32;
    blocksPerGrid3.x = ceil(double(NUM_VALUES) / double(threadsPerBlock3.x));
    blocksPerGrid3.y = ceil(double(NUM_VALUES) / double(threadsPerBlock3.y));
  }
  vectorProd<<<blocksPerGrid3, threadsPerBlock3, 0, stream>>>(d_a.get(), d_b.get(), d_ma.get(), NUM_VALUES);
  vectorProd<<<blocksPerGrid3, threadsPerBlock3, 0, stream>>>(d_a.get(), d_c.get(), d_mb.get(), NUM_VALUES);
  matrixMul<<<blocksPerGrid3, threadsPerBlock3, 0, stream>>>(d_ma.get(), d_mb.get(), d_mc.get(), NUM_VALUES);

  matrixMulVector<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_mc.get(), d_b.get(), d_c.get(), NUM_VALUES);

  cms::cuda::LogVerbatim("TestHeterogeneousEDProducerGPU")
      << "  " << label << " GPU kernels launched, returning return pointer device " << current_device << " CUDA stream "
      << stream;
  return d_a;
}

void TestCUDAProducerGPUKernel::runSimpleAlgo(float *d_data, cudaStream_t stream) const {
  int threadsPerBlock{32};
  int blocksPerGrid = (NUM_VALUES + threadsPerBlock - 1) / threadsPerBlock;
  vectorAddConstant<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_data, 1.0f, NUM_VALUES);
}
