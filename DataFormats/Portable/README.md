## Define portable data formats that wrap SoA data structures and can be persisted to ROOT files

### `PortableHostCollection<T>`

`PortableHostCollection<T>` is a class template that wraps a SoA type `T` and an alpaka host buffer, which owns the
memory where the SoA is allocated. The content of the SoA is persistent, while the buffer itself is transient.
Specialisations of this template can be persisted, and can be read back also in "bare ROOT" mode, without any
dictionaries.
They have no implicit or explicit references to alpaka (neither as part of the class signature nor as part of its name).
This could make it possible to read them back with different portability solutions in the future.

### `PortableDeviceCollection<T, TDev>`

`PortableDeviceCollection<T, TDev>` is a class template that wraps a SoA type `T` and an alpaka device buffer, which
owns the memory where the SoA is allocated.
To avoid confusion and ODR-violations, the `PortableDeviceCollection<T, TDev>` template cannot be used with the `Host`
device type.
Specialisations of this template are transient and cannot be persisted.

### `ALPAKA_ACCELERATOR_NAMESPACE::PortableCollection<T>`

`ALPAKA_ACCELERATOR_NAMESPACE::PortableCollection<T>` is a template alias that resolves to either
`PortableHostCollection<T>` or `PortableDeviceCollection<T, ALPAKA_ACCELERATOR_NAMESPACE::Device>`, depending on the
backend.

### `PortableCollection<T, TDev>`

`PortableCollection<T, TDev>` is an alias template that resolves to `ALPAKA_ACCELERATOR_NAMESPACE::PortableCollection<T>`
for the matching device.


## Notes

Modules that are supposed to work with only host types (_e.g._ dealing with de/serialisation, data transfers, _etc._)
should explicitly use the `PortableHostCollection<T>` types.

Modules that implement portable interfaces (_e.g._ producers) should use the generic types based on
`ALPAKA_ACCELERATOR_NAMESPACE::PortableCollection<T>` or `PortableCollection<T, TDev>`.
