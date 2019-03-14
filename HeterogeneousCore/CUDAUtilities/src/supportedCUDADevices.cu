#include <vector>

#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/supportedCUDADevices.h"

__global__
void isSupported(bool * result) {
  * result = true;
}

std::vector<int> supportedCUDADevices() {
  int devices = 0;
  auto status = cudaGetDeviceCount(&devices);
  if (status != cudaSuccess or devices == 0) {
    return {};
  }

  std::vector<int> supportedDevices;
  supportedDevices.reserve(devices);

  for (int i = 0; i < devices; ++i) {
    cudaCheck(cudaSetDevice(i));
    bool supported = false;
    bool * supported_d;
    cudaCheck(cudaMalloc(&supported_d, sizeof(bool)));
    cudaCheck(cudaMemset(supported_d, 0x00, sizeof(bool)));
    isSupported<<<1,1>>>(supported_d);
    // swallow any eventual error from launching the kernel on an unsupported device
    cudaGetLastError();
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaMemcpy(& supported, supported_d, sizeof(bool), cudaMemcpyDeviceToHost));
    cudaCheck(cudaFree(supported_d));
    if (supported) {
      supportedDevices.push_back(i);
    }
    cudaCheck(cudaDeviceReset());
  }

  return supportedDevices;
}
