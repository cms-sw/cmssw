# Introduction

The packages `HeterogeneousTest/ROCmDevice`, `HeterogeneousTest/ROCmKernel`,
`HeterogeneousTest/ROCmWrapper` and `HeterogeneousTest/ROCmOpaque` implement a set of libraries,
plugins and tests to exercise the build rules for ROCm.
In particular, these tests show what is supported and what are the limitations implementing
ROCm-based libraries, and using them from multiple plugins.


# `HeterogeneousTest/ROCmWrapper`

The package `HeterogeneousTest/ROCmWrapper` implements a library that defines and exports host-side
wrappers that launch the kernels defined in the `HeterogeneousTest/ROCmKernel` library:
```c++
namespace cms::cudatest {

  void wrapper_add_vectors_f(...);
  void wrapper_add_vectors_d(...);

}  // namespace cms::cudatest
```
These wrappers can be used from host-only, non-ROCm aware libraries, plugins and applications. They
can be linked with the standard host linker.

The `plugins` directory implements the `ROCmTestWrapperAdditionModule` `EDAnalyzer` that calls the
wrappers defined in this library. This plugin shows how the wrappers can be used directly from a
host-only, non-ROCm aware plugin.

The `test` directory implements the `testCudaDeviceAdditionWrapper` test binary that calls the
wrappers defined in this library, and shows how they can be used directly from a host-only, non-ROCm
aware application.
It also contains the `testROCmTestWrapperAdditionModule.py` python configuration to exercise the
`ROCmTestWrapperAdditionModule` module.


# Other packages

For various ways in which this library and plugin can be tested, see also the other
`HeterogeneousTest/ROCm...` packages:
  - [`HeterogeneousTest/ROCmDevice/README.md`](../../HeterogeneousTest/ROCmDevice/README.md)
  - [`HeterogeneousTest/ROCmKernel/README.md`](../../HeterogeneousTest/ROCmKernel/README.md)
  - [`HeterogeneousTest/ROCmOpaque/README.md`](../../HeterogeneousTest/ROCmOpaque/README.md)


# Combining plugins

`HeterogeneousTest/ROCmOpaque/test` contains the `testROCmTestAdditionModules.py` python
configuration that exercise all four plugins in a single application.
