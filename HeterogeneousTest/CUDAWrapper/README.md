# Introduction

The packages `HeterogeneousTest/CUDADevice`, `HeterogeneousTest/CUDAKernel`,
`HeterogeneousTest/CUDAWrapper` and `HeterogeneousTest/CUDAOpaque` implement a set of libraries,
plugins and tests to exercise the build rules for CUDA.
In particular, these tests show what is supported and what are the limitations implementing
CUDA-based libraries, and using them from multiple plugins.


# `HeterogeneousTest/CUDAWrapper`

The package `HeterogeneousTest/CUDAWrapper` implements a library that defines and exports host-side
wrappers that launch the kernels defined in the `HeterogeneousTest/CUDAKernel` library:
```c++
namespace cms::cudatest {

  void wrapper_add_vectors_f(...);
  void wrapper_add_vectors_d(...);

}  // namespace cms::cudatest
```
These wrappers can be used from host-only, non-CUDA aware libraries, plugins and applications. They
can be linked with the standard host linker.

The `plugins` directory implements the `CUDATestWrapperAdditionModule` `EDAnalyzer` that calls the
wrappers defined in this library. This plugin shows how the wrappers can be used directly from a
host-only, non-CUDA aware plugin.

The `test` directory implements the `testCudaDeviceAdditionWrapper` test binary that calls the
wrappers defined in this library, and shows how they can be used directly from a host-only, non-CUDA
aware application.
It also contains the `testCUDATestWrapperAdditionModule.py` python configuration to exercise the
`CUDATestWrapperAdditionModule` module.


# Other packages

For various ways in which this library and plugin can be tested, see also the other
`HeterogeneousTest/CUDA...` packages:
  - [`HeterogeneousTest/CUDADevice/README.md`](../../HeterogeneousTest/CUDADevice/README.md)
  - [`HeterogeneousTest/CUDAKernel/README.md`](../../HeterogeneousTest/CUDAKernel/README.md)
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
