# Introduction

The packages `HeterogeneousTest/CUDADevice`, `HeterogeneousTest/CUDAKernel`,
`HeterogeneousTest/CUDAWrapper` and `HeterogeneousTest/CUDAOpaque` implement a set of libraries,
plugins and tests to exercise the build rules for CUDA.
In particular, these tests show what is supported and what are the limitations implementing
CUDA-based libraries, and using them from multiple plugins.


# `HeterogeneousTest/CUDADevice`

The package `HeterogeneousTest/CUDADevice` implements a library that defines and exports CUDA
device-side functions:
```c++
namespace cms::cudatest {

  __device__ void add_vectors_f(...);
  __device__ void add_vectors_d(...);

}  // namespace cms::cudatest
```

The `plugins` directory implements the `CUDATestDeviceAdditionModule` `EDAnalyzer` that launches a
CUDA kernel using the functions defined in ths library. As a byproduct this plugin also shows how
to split an `EDAnalyzer` or other framework plugin into a host-only part (in a `.cc` file) and a
device part (in a `.cu` file).

The `test` directory implements the `testCudaDeviceAddition` binary that launches a CUDA kernel
using these functions.
It also contains the `testCUDATestDeviceAdditionModule.py` python configuration to exercise the
`CUDATestDeviceAdditionModule` plugin.


# Other packages

For various ways in which this library and plugin can be tested, see also the other
`HeterogeneousTest/CUDA...` packages:
  - [`HeterogeneousTest/CUDAKernel/README.md`](../../HeterogeneousTest/CUDAKernel/README.md)
  - [`HeterogeneousTest/CUDAWrapper/README.md`](../../HeterogeneousTest/CUDAWrapper/README.md)
  - [`HeterogeneousTest/CUDAOpaque/README.md`](../../HeterogeneousTest/CUDAOpaque/README.md)


# Combining plugins

`HeterogeneousTest/CUDAOpaque/test` contains the `testCUDATestAdditionModules.py` python
configuration that tries to exercise all four plugins in a single application.
Unfortunately, the CUDA kernels used in the `CUDATestDeviceAdditionModule` plugin and those used in
the `HeterogeneousTest/CUDAKernel` library run into some kind of conflict, leading to the error
```
HeterogeneousTest/CUDAKernel/plugins/CUDATestKernelAdditionAlgo.cu, line 17:
cudaCheck(cudaGetLastError());
cudaErrorInvalidDeviceFunction: invalid device function
```
Using together the other three plugins does work correctly.
