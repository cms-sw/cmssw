# Introduction

The packages `HeterogeneousTest/AlpakaDevice`, `HeterogeneousTest/AlpakaKernel`,
`HeterogeneousTest/AlpakaWrapper` and `HeterogeneousTest/AlpakaOpaque` implement a set of libraries,
plugins and tests to exercise the build rules for Alpaka.
In particular, these tests show what is supported and what are the limitations implementing
Alpaka-based libraries, and using them from multiple plugins.


# `HeterogeneousTest/AlpakaOpaque`

The package `HeterogeneousTest/AlpakaOpaque` implements a non-Alpaka aware library, with functions
that call the wrappers defined in the `HeterogeneousTest/AlpakaWrapper` library:
```c++
namespace ALPAKA_ACCELERATOR_NAMESPACE::test {

  void opaque_add_vectors_f(...);
  void opaque_add_vectors_d(...);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::test
```

The `plugins` directory implements the `AlpakaTestOpqaueAdditionModule` `EDAnalyzer` that calls the 
function defined in this library. This plugin shows how the function can be used directly from a 
host-only, non-Alpaka aware plugin.

The `test` directory implements the `testAlpakaDeviceAdditionOpqaue` test binary that calls the
function defined in this library, and shows how they can be used directly from a host-only,
non-Alpaka aware application.
It also contains the `testAlpakaTestOpqaueAdditionModule.py` python configuration to exercise the
`AlpakaTestOpqaueAdditionModule` module.


# Other packages

For various ways in which this library and plugin can be tested, see also the other
`HeterogeneousTest/Alpaka...` packages:
  - [`HeterogeneousTest/AlpakaDevice/README.md`](../../HeterogeneousTest/AlpakaDevice/README.md)
  - [`HeterogeneousTest/AlpakaKernel/README.md`](../../HeterogeneousTest/AlpakaKernel/README.md)
  - [`HeterogeneousTest/AlpakaWrapper/README.md`](../../HeterogeneousTest/AlpakaWrapper/README.md)


# Combining plugins

`HeterogeneousTest/AlpakaOpaque/test` contains also the `testAlpakaTestAdditionModules.py` python
configuration that exercise all four plugins in a single application.
