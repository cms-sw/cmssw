# Vertex CUDA Data Formats

`CUDADataFormat`s meant to be used on Host (CPU) or Device (CUDA GPU) for
storing information about vertices created during the Pixel-local Reconstruction
chain. It stores data in an SoA manner. It contains the data that was previously
contained in the deprecated `ZVertexSoA` class. 

The host format is inheriting from `CUDADataFormats/Common/interface/PortableHostCollection.h`,
while the device format is inheriting from `CUDADataFormats/Common/interface/PortableDeviceCollection.h`

Both formats use the same SoA Layout (`ZVertexSoAHeterogeneousLayout`) which is generated
via the `GENERATE_SOA_LAYOUT` macro in the `ZVertexUtilities.h` file.

## Notes

- Initially, `ZVertexSoA` had distinct array sizes for each attribute (e.g. `zv` was `MAXVTX` elements 
long, `ndof` was `MAXTRACKS` elements long). All columns are now of uniform `MAXTRACKS` size, 
meaning that there will be some wasted space (appx. 190kB). 
- Host and Device classes should **not** be created via inheritance, as they're done here,
but via composition. See [this discussion](https://github.com/cms-sw/cmssw/pull/40465#discussion_r1066039309).

## ZVertexHeterogeneousHost

The version of the data format to be used for storing vertex data on the CPU. 
Instances of this class are to be used for:

- Having a place to copy data to host from device, via `cudaMemcpy`, or
- Running host-side algorithms using data stored in an SoA manner.

## ZVertexHeterogeneousDevice

The version of the data format to be used for storing vertex data on the GPU.

Instances of `ZVertexHeterogeneousDevice` are to be created on host and be
used on device only. To do so, the instance's `view()` method is to be called
to pass a `View` to any kernel launched. Accessing data from the `view()` is not
possible on the host side.

## Utilities

Apart from `ZVertexSoAHeterogeneousLayout`, `ZVertexUtilities.h` also contains
a collection of methods which were originally
defined as class methods inside the `ZVertexSoA` class
which have been adapted to operate on `View` instances, so that they are callable
from within `__global__` kernels, on both CPU and CPU. 
