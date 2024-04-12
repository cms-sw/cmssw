# Introduction

The packages `HeterogeneousTest/AlpakaDevice`, `HeterogeneousTest/AlpakaKernel`,
`HeterogeneousTest/AlpakaWrapper` and `HeterogeneousTest/AlpakaOpaque` implement a set of libraries,
plugins and tests to exercise the build rules for Alpaka.
In particular, these tests show what is supported and what are the limitations implementing
Alpaka-based libraries, and using them from multiple plugins.


# `HeterogeneousTest/AlpakaDevice`

The package `HeterogeneousTest/AlpakaDevice` implements a library that defines and exports Alpaka
device-side functions:
```c++
namespace ALPAKA_ACCELERATOR_NAMESPACE::test {

  inline ALPAKA_FN_ACC void add_vectors_f(Acc1D const& acc, ...) { ... }

  inline ALPAKA_FN_ACC void add_vectors_d(Acc1D const& acc, ...) { ... }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::test
```

The `plugins` directory implements the `AlpakaTestDeviceAdditionModule` `EDAnalyzer` that launches
an Alpaka kernel using the functions defined in ths library. As a byproduct this plugin also shows
how to split an `EDAnalyzer` or other framework plugin into a host-only part (in a `.cc` file) and
a device part (in a `.dev.cc` file).

The `test` directory implements the `testAlpakaDeviceAddition` binary that launches a Alpaka kernel
using these functions.
It also contains the `testAlpakaTestDeviceAdditionModule.py` python configuration to exercise the
`AlpakaTestDeviceAdditionModule` plugin.


# Other packages

For various ways in which this library and plugin can be tested, see also the other
`HeterogeneousTest/Alpaka...` packages:
  - [`HeterogeneousTest/AlpakaKernel/README.md`](../../HeterogeneousTest/AlpakaKernel/README.md)
  - [`HeterogeneousTest/AlpakaWrapper/README.md`](../../HeterogeneousTest/AlpakaWrapper/README.md)
  - [`HeterogeneousTest/AlpakaOpaque/README.md`](../../HeterogeneousTest/AlpakaOpaque/README.md)


# Combining plugins

`HeterogeneousTest/AlpakaOpaque/test` contains the `testAlpakaTestAdditionModules.py` python
configuration that exercise all four plugins in a single application.
