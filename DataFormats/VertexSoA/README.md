# Vertex Portable Data Formats

`DataFormat`s meant to be used on Host (CPU) or Device (GPUs) for storing
information about reconstructed pixel vertices in Structure of Array (SoA)
format.


The `ZVertexBlocks` layout is composed of the layouts `ZVertexLayout` and `ZVertexTracksLayout`
using the `GENERATE_SOA_BLOCKS` macro. The `PortableHostCollection<reco::ZVertexBlocks>` and 
`PortableDeviceCollection<reco::ZVertexBlocks, TDev>` can be created using a different number of elements
for each sub-layout defined at run-time.
All Layouts are defined by the `GENERATE_SOA_LAYOUT` and `GENERATE_SOA_BLOCKS` macro in
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
