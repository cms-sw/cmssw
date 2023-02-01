# Introduction

The packages `HeterogeneousTest/ROCmDevice`, `HeterogeneousTest/ROCmKernel`,
`HeterogeneousTest/ROCmWrapper` and `HeterogeneousTest/ROCmOpaque` implement a set of libraries,
plugins and tests to exercise the build rules for ROCm.
In particular, these tests show what is supported and what are the limitations implementing
ROCm-based libraries, and using them from multiple plugins.


# `HeterogeneousTest/ROCmOpaque`

The package `HeterogeneousTest/ROCmOpaque` implements a non-ROCm aware library, with functions that
call the wrappers defined in the `HeterogeneousTest/ROCmWrapper` library:
```c++
namespace cms::cudatest {

  void opaque_add_vectors_f(...);
  void opaque_add_vectors_d(...);

}  // namespace cms::cudatest
```

The `plugins` directory implements the `ROCmTestOpqaueAdditionModule` `EDAnalyzer` that calls the 
function defined in this library. This plugin shows how the function can be used directly from a 
host-only, non-ROCm aware plugin.

The `test` directory implements the `testCudaDeviceAdditionOpqaue` test binary that calls the
function defined in this library, and shows how they can be used directly from a host-only, non-ROCm
aware application.
It also contains the `testROCmTestOpqaueAdditionModule.py` python configuration to exercise the
`ROCmTestOpqaueAdditionModule` module.


# Other packages

For various ways in which this library and plugin can be tested, see also the other
`HeterogeneousTest/ROCm...` packages:
  - [`HeterogeneousTest/ROCmDevice/README.md`](../../HeterogeneousTest/ROCmDevice/README.md)
  - [`HeterogeneousTest/ROCmKernel/README.md`](../../HeterogeneousTest/ROCmKernel/README.md)
  - [`HeterogeneousTest/ROCmWrapper/README.md`](../../HeterogeneousTest/ROCmWrapper/README.md)


# Combining plugins

`HeterogeneousTest/ROCmOpaque/test` contains also the `testROCmTestAdditionModules.py` python
configuration that exercise all four plugins in a single application.
