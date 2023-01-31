# Introduction

The packages `HeterogeneousTest/CUDADevice`, `HeterogeneousTest/CUDAKernel`,
`HeterogeneousTest/CUDAWrapper` and `HeterogeneousTest/CUDAOpaque` implement a set of libraries,
plugins and tests to exercise the build rules for CUDA.
In particular, these tests show what is supported and what are the limitations implementing
CUDA-based libraries, and using them from multiple plugins.


# `HeterogeneousTest/CUDAOpaque`

The package `HeterogeneousTest/CUDAOpaque` implements a non-CUDA aware library, with functions that
call the wrappers defined in the `HeterogeneousTest/CUDAWrapper` library:
```c++
namespace cms::cudatest {

  void opaque_add_vectors_f(...);
  void opaque_add_vectors_d(...);

}  // namespace cms::cudatest
```

The `plugins` directory implements the `CUDATestOpqaueAdditionModule` `EDAnalyzer` that calls the 
function defined in this library. This plugin shows how the function can be used directly from a 
host-only, non-CUDA aware plugin.

The `test` directory implements the `testCudaDeviceAdditionOpqaue` test binary that calls the
function defined in this library, and shows how they can be used directly from a host-only, non-CUDA
aware application.
It also contains the `testCUDATestOpqaueAdditionModule.py` python configuration to exercise the
`CUDATestOpqaueAdditionModule` module.


# Other packages

For various ways in which this library and plugin can be tested, see also the other
`HeterogeneousTest/CUDA...` packages:
  - [`HeterogeneousTest/CUDADevice/README.md`](../../HeterogeneousTest/CUDADevice/README.md)
  - [`HeterogeneousTest/CUDAKernel/README.md`](../../HeterogeneousTest/CUDAKernel/README.md)
  - [`HeterogeneousTest/CUDAWrapper/README.md`](../../HeterogeneousTest/CUDAWrapper/README.md)


# Combining plugins

`HeterogeneousTest/CUDAOpaque/test` contains also the `testCUDATestAdditionModules.py` python
configuration that tries to exercise all four plugins in a single application.
Unfortunately, the CUDA kernels used in the `CUDATestDeviceAdditionModule` plugin and those used in
the `HeterogeneousTest/CUDAKernel` library run into some kind of conflict, leading to the error
```
HeterogeneousTest/CUDAKernel/plugins/CUDATestKernelAdditionAlgo.cu, line 17:
cudaCheck(cudaGetLastError());
cudaErrorInvalidDeviceFunction: invalid device function
```
Using together the other three plugins does work correctly.
