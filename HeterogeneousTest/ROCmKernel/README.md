# Introduction

The packages `HeterogeneousTest/ROCmDevice`, `HeterogeneousTest/ROCmKernel`,
`HeterogeneousTest/ROCmWrapper` and `HeterogeneousTest/ROCmOpaque` implement a set of libraries,
plugins and tests to exercise the build rules for ROCm.
In particular, these tests show what is supported and what are the limitations implementing
ROCm-based libraries, and using them from multiple plugins.


# `HeterogeneousTest/ROCmKernel`

The package `HeterogeneousTest/ROCmKernel` implements a library that defines and exports ROCm
kernels that call the device functions defined in the `HeterogeneousTest/ROCmDevice` library:
```c++
namespace cms::cudatest {

  __global__ void kernel_add_vectors_f(...);
  __global__ void kernel_add_vectors_d(...);

}  // namespace cms::cudatest
```

The `plugins` directory implements the `ROCmTestKernelAdditionModule` `EDAnalyzer` that launches the
ROCm kernels defined in this library. As a byproduct this plugin also shows how to split an
`EDAnalyzer` or other framework plugin into a host-only part (in a `.cc` file) and a device part (in
a `.cu` file).

The `test` directory implements the `testCudaKernelAddition` test binary that launches the ROCm kernel
defined in this library.
It also contains the `testROCmTestKernelAdditionModule.py` python configuration to exercise the
`ROCmTestKernelAdditionModule` module.


# Other packages

For various ways in which this library and plugin can be tested, see also the other
`HeterogeneousTest/ROCm...` packages:
  - [`HeterogeneousTest/ROCmDevice/README.md`](../../HeterogeneousTest/ROCmDevice/README.md)
  - [`HeterogeneousTest/ROCmWrapper/README.md`](../../HeterogeneousTest/ROCmWrapper/README.md)
  - [`HeterogeneousTest/ROCmOpaque/README.md`](../../HeterogeneousTest/ROCmOpaque/README.md)


# Combining plugins

`HeterogeneousTest/ROCmOpaque/test` contains the `testROCmTestAdditionModules.py` python
configuration that exercise all four plugins in a single application.
