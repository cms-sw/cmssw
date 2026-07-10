# `DataFormat`-safe alpaka types

This package contains the types that are used in the implementation of the
alpaka-based heterogeneous framework that do not depend on the execution aspects
like the framework `Service` and `Async` interfaces, and can thus be included in
other `DataFormats` packages.

Currently this package defines

  - `EDMetadata`, that implements the synchronization mechanisms for `Event`
    data products for asynchronous backends;

  - `DeviceProductType<T>`, that conditionally resolves to `T` (for synchronous
    backends) or `edm::DeviceProduct<T>` (for asynchronous ones).
