# Introduction

The packages `HeterogeneousTest/AlpakaDevice`, `HeterogeneousTest/AlpakaKernel`,
`HeterogeneousTest/AlpakaWrapper` and `HeterogeneousTest/AlpakaOpaque` implement a set of libraries,
plugins and tests to exercise the build rules for Alpaka.
In particular, these tests show what is supported and what are the limitations implementing
Alpaka-based libraries, and using them from multiple plugins.


# `HeterogeneousTest/AlpakaWrapper`

The package `HeterogeneousTest/AlpakaWrapper` implements a library that defines and exports
host-side wrappers that launch the kernels defined in the `HeterogeneousTest/AlpakaKernel` library:
```c++
namespace ALPAKA_ACCELERATOR_NAMESPACE::test {

  void wrapper_add_vectors_f(...);
  void wrapper_add_vectors_d(...);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::test
```
These wrappers can be used from host-only, non-Alpaka aware libraries, plugins and applications.
They can be linked with the standard host linker.

The `plugins` directory implements the `AlpakaTestWrapperAdditionModule` `EDAnalyzer` that calls the
wrappers defined in this library. This plugin shows how the wrappers can be used directly from a
host-only, non-Alpaka aware plugin.

The `test` directory implements the `testAlpakaDeviceAdditionWrapper` test binary that calls the
wrappers defined in this library, and shows how they can be used directly from a host-only,
non-Alpaka aware application.
It also contains the `testAlpakaTestWrapperAdditionModule.py` python configuration to exercise the
`AlpakaTestWrapperAdditionModule` module.


# Other packages

For various ways in which this library and plugin can be tested, see also the other
`HeterogeneousTest/Alpaka...` packages:
  - [`HeterogeneousTest/AlpakaDevice/README.md`](../../HeterogeneousTest/AlpakaDevice/README.md)
  - [`HeterogeneousTest/AlpakaKernel/README.md`](../../HeterogeneousTest/AlpakaKernel/README.md)
  - [`HeterogeneousTest/AlpakaOpaque/README.md`](../../HeterogeneousTest/AlpakaOpaque/README.md)


# Combining plugins

`HeterogeneousTest/AlpakaOpaque/test` contains the `testAlpakaTestAdditionModules.py` python
configuration that exercise all four plugins in a single application.
