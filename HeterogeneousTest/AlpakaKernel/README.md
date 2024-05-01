# Introduction

The packages `HeterogeneousTest/AlpakaDevice`, `HeterogeneousTest/AlpakaKernel`,
`HeterogeneousTest/AlpakaWrapper` and `HeterogeneousTest/AlpakaOpaque` implement a set of libraries,
plugins and tests to exercise the build rules for Alpaka.
In particular, these tests show what is supported and what are the limitations implementing
Alpaka-based libraries, and using them from multiple plugins.


# `HeterogeneousTest/AlpakaKernel`

The package `HeterogeneousTest/AlpakaKernel` implements a library that defines and exports Alpaka
kernels that call the device functions defined in the `HeterogeneousTest/AlpakaDevice` library:
```c++
namespace ALPAKA_ACCELERATOR_NAMESPACE::test {

  struct KernelAddVectorsF {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, ...) const { ... }
  };

  struct KernelAddVectorsD {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, ...) const { ... }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::test
```

The `plugins` directory implements the `AlpakaTestKernelAdditionModule` `EDAnalyzer` that launches
the Alpaka kernels defined in this library. As a byproduct this plugin also shows how to split an
`EDAnalyzer` or other framework plugin into a host-only part (in a `.cc` file) and a device part (in
a `.dev.cc` file).

The `test` directory implements the `testAlpakaKernelAddition` test binary that launches the Alpaka
kernel defined in this library.
It also contains the `testAlpakaTestKernelAdditionModule.py` python configuration to exercise the
`AlpakaTestKernelAdditionModule` module.


# Other packages

For various ways in which this library and plugin can be tested, see also the other
`HeterogeneousTest/Alpaka...` packages:
  - [`HeterogeneousTest/AlpakaDevice/README.md`](../../HeterogeneousTest/AlpakaDevice/README.md)
  - [`HeterogeneousTest/AlpakaWrapper/README.md`](../../HeterogeneousTest/AlpakaWrapper/README.md)
  - [`HeterogeneousTest/AlpakaOpaque/README.md`](../../HeterogeneousTest/AlpakaOpaque/README.md)


# Combining plugins

`HeterogeneousTest/AlpakaOpaque/test` contains the `testAlpakaTestAdditionModules.py` python
configuration that exercise all four plugins in a single application.
