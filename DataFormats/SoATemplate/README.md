# Structure of array (SoA) generation

The header file [`SoALayout.h`](SoALayout.h) defines preprocessor macros that
allow generating SoA classes. The SoA classes generate multiple, aligned columns from a memory buffer. The memory
buffer is allocated separately by the user, and can be located in a memory space different from the local one (for
example, a SoA located in a GPU device memory can be fully pre-defined on the host and the resulting structure is
passed to the GPU kernel).

This columnar storage allows efficient memory access by GPU kernels (coalesced access on cache line aligned data)
and possibly vectorization.

Additionally, templation of the layout and view classes allows compile-time variations of accesses and checks:
verification of alignment and corresponding compiler hinting, cache strategy (non-coherent, streaming with immediate
invalidation), range checking.

The implementation relies on the Boost Preprocessor library to generate the required boilerplate. 
This approach keeps the user-facing code concise while providing a natural (AoS-like) interface for accessing SoA data.

## Layout

`SoALayout` is a macro-generated templated class that subdivides a provided buffer into a collection of columns,
Eigen columns and scalars. The buffer is expected to be aligned with a selectable alignment in bytes. See [Template
parameters section](#template-parameters) for more information. All columns and scalars within a `SoALayout` will 
be individually aligned, leaving padding at the end of each if necessary. Eigen columns have each component of 
the vector or matrix properly aligned in individual columns (by defining the stride between components).
Only compile-time sized Eigen vectors and matrices are supported. Scalar members are members of the layout with one
element, irrespective of the size of the layout.

Static utility functions automatically compute the byte size of a layout, taking into account all its columns and
alignment.

## View

Layout classes also define a `View` and `ConstView` subclass that provide access to each column and
scalar of the layout. In addition to those fully parametrized templates, two other levels of parametrization are
provided: `ViewTemplate`, `ViewViewTemplateFreeParams` and respectively `ConstViewTemplate`,
`ConstViewTemplateFreeParams`. The parametrization of those templates is explained in the [Template
parameters section](#template-parameters).

The view can be generated in constant (`ConstView`) and non-constant (`View`) flavors. All view flavors provide the 
same interface where scalar elements are accessed with an `operator()`: `soa.scalar()` while columns (Eigen or not) are
accessed via an array of structure (AoS)-like syntax: `soa[index].x()`. The proxy object returned by `operator[]`
can be stored and reused as a convenient shorthand: `auto si = soa[index]; si.z() = si.x() + si.y();`. It is also 
possible to access the data in a more SoA-natural way: `soa.x()[index]` or `soa.x(index)`.

A view can be constructed either from the corresponding layout or from the Metarecords subclass of other layouts. 
Since a view is non-owning, its columns may refer to data belonging to different SoAs constructed from different 
memory buffers. Consequently, the columns referenced by a view are not required to be contiguous in memory.

## Descriptor

The nested class `ConstDescriptor` can only be instantiated by passing a `View` or a `ConstView`.
It provides access to columns and related information. This class should be considered an internal 
implementation detail, used solely by the SoA and EDM frameworks for performing heterogeneous memory operations. 
It is used to implement the `deepCopy` from a `View` referencing different memory buffers, as shown in 
[`PortableHostCollection<T>`](../../DataFormats/Portable/README.md#portablehostCollection)
and [`PortableDeviceCollection<T, TDev>`](../../DataFormats/Portable/README.md#portabledeviceCollection) sections.
Specifically, it exposes:
- the columns as an `std::tuple<std::span<T>...>` accessible via `descriptor.buff`
- the corresponding column types as an `std::tuple<cms::soa::SoAColumnType>` through `descriptor.columnTypes`.

## Metadata subclass

To avoid cluttering the namespace of the generated layout class, a subclass called `Metadata` is generated. It is
instantiated with the `metadata()` member function and provides information about the layout and 
its underlying storage, including:

- `size()`: The number of elements per column in the SoA
- `byteSize()`: The total size of the buffer required by the layout
- `alignment()`: The alignment in bytes applied to each column
- `data()`: Returns a pointer to the `std::byte` buffer of the layout
- `nextByte()`: Returns the next byte after a layout, used for creating multiple layouts from a single buffer
- `cloneToNewAddress()`: Creates a new layout using a new buffer but the same number of elements per column

## Metarecords subclass

The nested type `Metarecords` describes the elements of the SoA. It can be instantiated by the `records()` member 
function of a `View` or `ConstView`. Every object contains the address of the first element of the column, the number
of elements per column, and the stride for the Eigen columns. These are used to validate the column size at run time 
and to build a generic `View` as described in [View](#view).

## Customized methods

It is possible to generate methods inside the `element` and `const_element` nested structs using the 
`SOA_ELEMENT_METHODS` and `SOA_CONST_ELEMENT_METHODS` macros. Each of these macros can be called only once, 
and can define multiple methods. Note that `SOA_ELEMENT_METHODS` and `SOA_CONST_ELEMENT_METHODS` should be prefixed 
with the macro SOA_HOST_DEVICE.  This ensures that the methods can also be executed in device kernels.
[An example is shown below.](#examples)

## Blocks

`SoABlocks` is a macro-generated templated class that enables structured composition of multiple `SoALayouts` 
into a single container, referred to as "blocks". Each block is a Layout, and the structure itself 
looks like multiple contiguous memory buffers of different sizes. 
The block of an `SoABlock` layout can be an `SoABlock` in itself. Like this, nested SoA-layouts can be created. 
Classes generated by the `GENERATE_SOA_BLOCKS` macro have the same template arguments as normal SoA-layouts. 
The template arguments are passed to each block to ensure that, for example, the alignment is the same for every block. 
`SoABlocks` also supports `View` and `ConstView` classes. 
In addition to those fully parametrized templates, two further levels of parametrization are provided:
`ViewTemplate`, `ViewTemplateFreeParams` and respectively `ConstViewTemplate`, `ConstViewTemplateFreeParams`, 
mirroring the structure of the underlying structs. The blocks are built via composition, 
and access to individual layouts and views is provided by name.

`SoABlocks` also have the possibility of generating methods for the `View` and `ConstView` classes 
using the macros `SOA_VIEW_METHODS` and `SOA_CONST_VIEW_METHODS`. 
Like the macros for the element methods, this can also be called only once, and if more methods
have to be generated, they must be listed inside the same macro call. Since these methods can be called from the device,
they must be prefixed with the `SOA_HOST_DEVICE` macro and, when possible, with the `constexpr` keyword.

[An example of utilization is shown below.](#examples)

## ROOT serialization and de-serialization

Layouts can be serialized and de-serialized with ROOT. To generate the ROOT dictionary, separate
`clases_def.xml` and `classes.h` should be prepared. `classes.h` ensures the inclusion of the proper header files to
get the definition of the serialized classes, and `classes_def.xml` needs to define the fixed list of members that
ROOT should ignore, plus the list of all the columns. [An example is provided below.](#examples)

## Template parameters

The template arguments of the generated SoA-layouts are:
- `ALIGNMENT` (default: 128 bytes): The byte alignment of each column, Eigen column, and scalar. 
  While the optimal alignment depends on the target hardware, using the same alignment across all devices in a 
  heterogeneous environment is generally preferable. This ensures that the entire backing buffer has the same memory 
  layout everywhere, allowing it to be transferred between devices in a single operation. If different alignments 
  are used, each column must instead be transferred individually.
- `ALIGNMENT_ENFORCEMENT` (default: `relaxed`): When enforced, the alignment of the whole buffer will be 
  checked at construction time of the layout, and the alignment of each column will be checked at construction
  time of a view. Possible arguments are `enforced` (true) or `relaxed` (false)

The template arguments of the Views are:
- `RESTRICT_QUALIFY` (default: true): 
  Adds `__restrict__` qualifiers to the column pointers, allowing the compiler
  to assume that they do not alias. This enables more aggressive optimizations like SIMD vectorisation, or 
  for example on NVIDIA GPUs it results in the generation of load instructions that use the faster non-coherent cache.
- `RANGE_CHECKING` (default: `cms::soa::RangeChecking::Default`): 
  Adds out-of-bounds index checking on each access at runtime when using `enabled` or `extended`. `extended` 
  additionally outputs the file and line number of where `[]-operator` was called with a faulty index. 
  This is achieved by using `std::source_location`. As this is a compile-time parameter, the cost of the feature at
  run time is null if turned off. When turned on, the accesses will be slowed down by checks. Upon error detection,
  an exception is launched (on the CPU side) or the kernel is made to crash (on the GPU side). This feature can help
  the debugging of index issues at runtime, but of course requires a recompilation.

Several predefined view types are generated with different levels of template parameterization:
- `View`: uses the same template for `ALIGNMENT` and `ALIGNMENT_ENFORCEMENT` as the corresponding layout, 
  while using the default settings for `RESTRICT_QUALIFY` and `RANGE_CHECKING`.
- `ViewTemplate`: additionally exposes `RESTRICT_QUALIFY` and `RANGE_CHECKING`.
- `ViewTemplateFreeParams`: exposes all template parameters, allowing complete customization of the view.

Note that the same variants for const access are available through 
`ConstView`, `ConstViewTemplate`, `ConstViewTemplateFreeParams`.
Views are lightweight, trivially copyable objects. Consequently, 
converting between views with different template parameters is inexpensive.

## Using SoA layouts and views with GPUs

An SoA layout is a host-side object and cannot be used directly inside a GPU kernel. 
A view, on the other hand, is a lightweight object containing only one pointer per column and 
the total number of elements. Views are typically constructed on the host and passed to GPU kernels by value, 
although they can also be constructed on the device if needed.

Additional view types can be generated that expose only a selected subset of columns, 
reducing the number of stored pointers for kernels that access only part of the SoA.

## Examples

A layout can be defined as:

```C++
#include "DataFormats/SoALayout.h"

GENERATE_SOA_LAYOUT(SoA1LayoutTemplate,
  // Columns: one value per SoA element. The element type may be a
  // fundamental type, struct, or class.
  SOA_COLUMN(double, x),
  SOA_COLUMN(double, y),
  SOA_COLUMN(double, z),
  SOA_COLUMN(uint16_t, color),
  SOA_COLUMN(int32_t, value),
  SOA_COLUMN(double *, py),
  SOA_COLUMN(uint32_t, count),
  SOA_COLUMN(uint32_t, anotherCount),

  // Eigen columns: fixed-size Eigen vectors or matrices stored in a
  // columnar layout using one SoA column per component.
  SOA_EIGEN_COLUMN(Eigen::Vector3d, a),
  SOA_EIGEN_COLUMN(Eigen::Vector3d, b),
  SOA_EIGEN_COLUMN(Eigen::Vector3d, r),

  // Scalars: a single value shared by the entire SoA, independent of
  // the number of elements.
  SOA_SCALAR(const char *, description),
  SOA_SCALAR(uint32_t, someNumber)
);

// Default template parameters are <
//   size_t ALIGNMENT = cms::soa::CacheLineSize::defaultSize,
//   bool ALIGNMENT_ENFORCEMENT = cms::soa::AlignmentEnforcement::relaxed
// >
using SoA1Layout = SoA1LayoutTemplate<>;

using SoA1LayoutAligned = SoA1LayoutTemplate<cms::soa::CacheLineSize::defaultSize, 
                                             cms::soa::AlignmentEnforcement::enforced>;
```

It is possible to declare methods that operate on the SoA elements:

```C++
#include "DataFormats/SoALayout.h"

GENERATE_SOA_LAYOUT(SoATemplate,
  SOA_COLUMN(double, x),
  SOA_COLUMN(double, y),
  SOA_COLUMN(double, z),
  
  // methods operating on const_element
  SOA_CONST_ELEMENT_METHODS(
    SOA_HOST_DEVICE auto norm() const {
      return sqrt(x()*x() + y()+y() + z()*z());
    }
  ),

  // methods operating on element
  SOA_ELEMENT_METHODS(
    SOA_HOST_DEVICE void scale(float arg) {
      x() *= arg;
      y() *= arg;
      z() *= arg;
    }
  ),
  
  SOA_SCALAR(int, detectorType)
);

using SoA = SoATemplate<>;
using SoAView = SoA::View;
using SoAConstView = SoA::ConstView;
```

as well as methods that operate on the View:

```C++
GENERATE_SOA_LAYOUT(PositionLayout,
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z))

GENERATE_SOA_LAYOUT(VelocityLayout,
                    SOA_COLUMN(float, vx),
                    SOA_COLUMN(float, vy),
                    SOA_COLUMN(float, vz))

GENERATE_SOA_BLOCKS(PointsLayout,
                    SOA_BLOCK(position, PositionLayout),
                    SOA_BLOCK(velocity, VelocityLayout),
                    SOA_VIEW_METHODS(
                        SOA_HOST_DEVICE void update_position(uint32_t i, float time) {
                            auto pos = this->position()[i];
                            auto vel = this->velocity()[i];
                            pos.x() += vel.vx() * time;
                            pos.y() += vel.vy() * time;
                            pos.z() += vel.vz() * time;
                        }
                    ),
                    SOA_CONST_VIEW_METHODS(
                        SOA_HOST_DEVICE auto distance2(uint32_t i, uint32_t j) const {
                            auto pi = this->position()[i];
                            auto pj = this->position()[j];
                            return (pi.x() - pj.x()) * (pi.x() - pj.x()) + 
                                   (pi.y() - pj.y()) * (pi.y() - pj.y()) + 
                                   (pi.z() - pj.z()) * (pi.z() - pj.z());
                        }
                    )
)
```

The buffer of the proper size is allocated, and the layout is populated with:

```C++
// Allocation of aligned
size_t elements = 100;
using AlignedBuffer = std::unique_ptr<std::byte, decltype(std::free) *>;
AlignedBuffer h_buf (reinterpret_cast<std::byte*>(aligned_alloc(SoA1LayoutAligned::alignment, 
                                                  SoA1LayoutAligned::computeDataSize(elements))), 
                                                  std::free);
SoA1LayoutAligned soaLayout(h_buf.get(), elements);
```

The SoA provides an overloaded operator<< that outputs a detailed representation of its memory layout, 
including column offsets, sizes, padding, and scalar fields. 
This facilitates inspection and verification of the SoA layout through standard C++ output streams.

```C++
// Introspection of an SoALayout
std::cout << soaLayout
```

The mutable and const views with the same set of columns and their 
parametrized variants are provided from the layout as:

```C++
// (Pseudo-code)
struct SoA1Layout::View;

template<bool RESTRICT_QUALIFY = cms::soa::RestrictQualify::enabled,
         bool RANGE_CHECKING = cms::soa::RangeChecking::enabled>
struct SoA1Layout::ViewTemplate;

template<size_t ALIGNMENT = cms::soa::CacheLineSize::defaultSize,
         bool ALIGNMENT_ENFORCEMENT = cms::soa::AlignmentEnforcement::relaxed,
         bool RESTRICT_QUALIFY = cms::soa::RestrictQualify::enabled,
         bool RANGE_CHECKING = cms::soa::RangeChecking::enabled>
struct SoA1Layout::ViewTemplateFreeParams;

struct SoA1Layout::ConstView;

template<bool RESTRICT_QUALIFY = cms::soa::RestrictQualify::enabled,
         bool RANGE_CHECKING = cms::soa::RangeChecking::enabled>
struct SoA1Layout::ConstViewTemplate;

template<size_t ALIGNMENT = cms::soa::CacheLineSize::defaultSize,
         bool ALIGNMENT_ENFORCEMENT = cms::soa::AlignmentEnforcement::relaxed,
         bool RESTRICT_QUALIFY = cms::soa::RestrictQualify::enabled,
         bool RANGE_CHECKING = cms::soa::RangeChecking::enabled>
struct SoA1Layout::ConstViewTemplateFreeParams;
```

The generated `View` and `ConstView` types use compiler-specific non-aliasing annotations 
(such as `__restrict__`, where supported) and provide optional range checking.
`View` and `ConstView` are distinct types, with `View` inheriting from `ConstView`. 
Consequently, a `View` can be implicitly converted to a `ConstView`, while preserving const correctness.

Range checking can also be enabled in an extended mode with `cms::soa::RangeChecking::extended` 
as the template parameter of a view. In this mode, `std::source_location` is captured whenever an index is passed 
to a view, allowing out-of-bounds errors to report the originating file name and line number. 
This additional context makes it easier to identify the exact access responsible for an out-of-bounds error, 
improving the debugging experience. Views with the extended range checking can be created like this: 

```C++
// default views with cms::soa::RestrictQualify::Default and cms::soa::RangeChecking::Default
using SoAView = SoA::View;
using SoAConstView = SoA::ConstView;

// extended range checking to get more output when access out-of-bounds is encountered
using SoAViewExtended = SoA::ViewTemplate<cms::soa::RestrictQualify::Default, 
                                          cms::soa::RangeChecking::extended>;
using SoAConstViewExtended = SoA::ConstViewTemplate<cms::soa::RestrictQualify::Default, 
                                                    cms::soa::RangeChecking::extended>;
```

The SoA by blocks can be created in this way:

```C++
GENERATE_SOA_LAYOUT(SoAPositionTemplate,
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z),
                    SOA_SCALAR(int, detectorType))

GENERATE_SOA_LAYOUT(SoAPCATemplate,
                    SOA_COLUMN(float, eigenvector_1),
                    SOA_COLUMN(float, eigenvector_2),
                    SOA_COLUMN(float, eigenvector_3),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, candidateDirection))

GENERATE_SOA_LAYOUT(SoATemplate,
                    SOA_SCALAR(int, id),
                    SOA_SCALAR(int, type),
                    SOA_SCALAR(float, energy))

GENERATE_SOA_BLOCKS(SoABlocksTemplate,
                    SOA_BLOCK(position, SoAPositionTemplate),
                    SOA_BLOCK(pca, SoAPCATemplate),
                    SOA_BLOCK(scalars, SoATemplate))

using SoABlocks = SoABlocksTemplate<>;
using SoABlocksView = SoABlocks::View;
using SoABlocksConstView = SoABlocks::ConstView;                      
```                   

and the corresponding Views/ConstViews can be accessed like this:

```C++
// Create a SoABlocks instance with three blocks of different sizes
std::array<cms::soa::size_type, 3> sizes{{10, 20, 1}};
const std::size_t blocksBufferSize = SoABlocks::computeDataSize(sizes);

std::unique_ptr<std::byte, decltype(std::free) *> buffer{
    reinterpret_cast<std::byte *>(aligned_alloc(SoABlocks::alignment, blocksBufferSize)), std::free};

SoABlocks blocks(buffer.get(), sizes);    
SoABlocksView blocksView{blocks};
SoABlocksConstView blocksConstView{blocks};

// Fill the blocks with some data
blocksView.position().detectorType() = 1;
for (int i = 0; i < blocksView.position().metadata().size(); ++i) {
    blocksView.position()[i] = { 0.1f, 0.2f, 0.3f };
}
for (int i = 0; i < blocksView.metadata().size()[1]; ++i) {
    blocksView.pca()[i].eigenvector_1() = 0.0f;
    blocksView.pca()[i].eigenvector_2() = 0.0f;
    blocksView.pca()[i].eigenvector_3() = 1.0f;
    blocksView.pca()[i].candidateDirection() = Eigen::Vector3d(1.0, 0.0, 0.0);
}
blocksView.scalars().id() = 42;
blocksView.scalars().type() = 1;
blocksView.scalars().energy() = 100.0f;

// SoALayouts support introspection.
// Outputs all blocks contained in the SoABlocks layout,
// including the size of each column in bytes and its associated padding.
std::cout << blocks;
```
                   
## Current status and further improvements

### Available features

- The layout and views support scalars and columns, alignment and alignment enforcement, and hinting (linked).
- Automatic `__restrict__` compiler hinting is supported and can be enabled where appropriate.
- Automatic creation of trivial views and const views derived from a single layout.
- Cache access style, which was explored, was abandoned as this not-yet-used feature interferes with `__restrict__`
  support (which is already in use in existing code). It could be made available as a separate tool that can be used
  directly by the module developer, orthogonally from SoA.
- Optional (compile-time) range checking validates the index of every column access, throwing an exception on the
  CPU side and forcing a segmentation fault to halt kernels. When not enabled, it has no impact on performance (code
  not compiled). Using `RangeChecking::extended` causes a capture of the source location using `std::source_location`,
  when an integer index is passed to access the data. When an out-of-bounds error is thrown, 
  this leads to more information in the error message, including the file name and line number 
  where the out-of-bounds index was passed to the SoA.
- Eigen columns are also supported, with both const and non-const flavors.
- ROOT serialization and deserialization is supported. In CMSSW, it is planned to be used through the memory
  managing `PortableCollection` family of classes.
- An `operator<<()` is provided to print the layout of an SoA to standard streams.
