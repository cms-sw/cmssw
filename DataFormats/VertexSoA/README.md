# Vertex Portable Data Formats

`DataFormat`s meant to be used on Host (CPU) or Device (GPUs) for storing
information about reconstructed pixel vertices in Structure of Array (SoA)
format.

The host collection is an instantiation of `PortableHostMultiCollection`, while
the device collection is an instantiation of `PortableDeviceMultiCollection`.

Both collections use two SoA layouts (`ZVertexLayout` and `ZVertexTracksLayout`)
with different number of elements, defined at run-time.
The layouts are defined by the `GENERATE_SOA_LAYOUT` macro in
`DataFormats/VertexSoA/interface/ZVertexSoA.h`.


## `ZVertexHost`

The version of the data format to be used for storing vertex data on the CPU. 
Instances of this class are to be used for:
  - having a place to copy data to host from device, which is usually taken care
    of automatically by the framework;
  - running host-side algorithms using data stored in an SoA manner.


## `ZVertexDevice`

The version of the data format to be used for storing vertex data on the GPU.
Instances of `ZVertexDevice` are created on the host, and can only be used only
on the device. To do so, the instance's `view()` or `const_view()` methods are
called, and the resulting `View` or `ConstView` are passed to a kernel launch.
The data from the instance, `view()` or `const_view()` is not accessible on the
host side.
