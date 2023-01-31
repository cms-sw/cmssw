# Introduction

The packages `HeterogeneousTest/ROCmDevice`, `HeterogeneousTest/ROCmKernel`,
`HeterogeneousTest/ROCmWrapper` and `HeterogeneousTest/ROCmOpaque` implement a set of libraries,
plugins and tests to exercise the build rules for ROCm.
In particular, these tests show what is supported and what are the limitations implementing
ROCm-based libraries, and using them from multiple plugins.


# `HeterogeneousTest/ROCmDevice`

The package `HeterogeneousTest/ROCmDevice` implements a library that defines and exports ROCm
device-side functions:
```c++
namespace cms::cudatest {

  __device__ void add_vectors_f(...);
  __device__ void add_vectors_d(...);

}  // namespace cms::cudatest
```

The `plugins` directory implements the `ROCmTestDeviceAdditionModule` `EDAnalyzer` that launches a
ROCm kernel using the functions defined in ths library. As a byproduct this plugin also shows how
to split an `EDAnalyzer` or other framework plugin into a host-only part (in a `.cc` file) and a
device part (in a `.cu` file).

The `test` directory implements the `testCudaDeviceAddition` binary that launches a ROCm kernel
using these functions.
It also contains the `testROCmTestDeviceAdditionModule.py` python configuration to exercise the
`ROCmTestDeviceAdditionModule` plugin.


# Other packages

For various ways in which this library and plugin can be tested, see also the other
`HeterogeneousTest/ROCm...` packages:
  - [`HeterogeneousTest/ROCmKernel/README.md`](../../HeterogeneousTest/ROCmKernel/README.md)
  - [`HeterogeneousTest/ROCmWrapper/README.md`](../../HeterogeneousTest/ROCmWrapper/README.md)
  - [`HeterogeneousTest/ROCmOpaque/README.md`](../../HeterogeneousTest/ROCmOpaque/README.md)


# Combining plugins

`HeterogeneousTest/ROCmOpaque/test` contains the `testROCmTestAdditionModules.py` python
configuration that exercise all four plugins in a single application.
