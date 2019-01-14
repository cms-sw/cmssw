#ifndef HeterogeneousCore_CUDAServices_numberOfCUDADevices_h
#define HeterogeneousCore_CUDAServices_numberOfCUDADevices_h

// Returns the number of CUDA devices
// The difference wrt. the standard CUDA function is that if
// CUDAService is disabled, this function returns 0.
int numberOfCUDADevices();

#endif
