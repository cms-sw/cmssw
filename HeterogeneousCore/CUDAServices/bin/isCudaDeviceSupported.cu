#include <cuda_runtime.h>

#include "isCudaDeviceSupported.h"

__global__ static void setSupported(bool* result) { *result = true; }

bool isCudaDeviceSupported(int device) {
  bool supported = false;
  bool* supported_d;

  // select the requested device - will fail if the index is invalid
  cudaError_t status = cudaSetDevice(device);
  if (status != cudaSuccess)
    return false;

  // allocate memory for the flag on the device
  status = cudaMalloc(&supported_d, sizeof(bool));
  if (status != cudaSuccess)
    return false;

  // initialise the flag on the device
  status = cudaMemset(supported_d, 0x00, sizeof(bool));
  if (status != cudaSuccess)
    return false;

  // try to set the flag on the device
  setSupported<<<1, 1>>>(supported_d);

  // check for an eventual error from launching the kernel on an unsupported device
  status = cudaGetLastError();
  if (status != cudaSuccess)
    return false;

  // wait for the kernelto run
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess)
    return false;

  // copy the flag back to the host
  status = cudaMemcpy(&supported, supported_d, sizeof(bool), cudaMemcpyDeviceToHost);
  if (status != cudaSuccess)
    return false;

  // free the device memory
  status = cudaFree(supported_d);
  if (status != cudaSuccess)
    return false;

  // reset the device
  status = cudaDeviceReset();
  if (status != cudaSuccess)
    return false;

  return supported;
}
