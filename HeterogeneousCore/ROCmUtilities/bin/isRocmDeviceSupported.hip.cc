#include <hip/hip_runtime.h>

#include "isRocmDeviceSupported.h"

namespace {
  __global__ void setSupported(bool* result) { *result = true; }
}  // namespace

bool isRocmDeviceSupported(int device) {
  bool supported = false;
  bool* supported_d;

  // select the requested device - will fail if the index is invalid
  hipError_t status = hipSetDevice(device);
  if (status != hipSuccess)
    return false;

  // allocate memory for the flag on the device
  status = hipMalloc(&supported_d, sizeof(bool));
  if (status != hipSuccess)
    return false;

  // initialise the flag on the device
  status = hipMemset(supported_d, 0x00, sizeof(bool));
  if (status != hipSuccess)
    return false;

  // try to set the flag on the device
  setSupported<<<1, 1>>>(supported_d);

  // check for an eventual error from launching the kernel on an unsupported device
  status = hipGetLastError();
  if (status != hipSuccess)
    return false;

  // wait for the kernelto run
  status = hipDeviceSynchronize();
  if (status != hipSuccess)
    return false;

  // copy the flag back to the host
  status = hipMemcpy(&supported, supported_d, sizeof(bool), hipMemcpyDeviceToHost);
  if (status != hipSuccess)
    return false;

  // free the device memory
  status = hipFree(supported_d);
  if (status != hipSuccess)
    return false;

  // reset the device
  status = hipDeviceReset();
  if (status != hipSuccess)
    return false;

  return supported;
}
