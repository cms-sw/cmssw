# Structure of array (SoA) generation

The header file [`SoALayout.h`](SoALayout.h) defines preprocessor macros that
allow generating SoA classes. The SoA classes generate multiple, aligned column from a memory buffer. The memory
buffer is allocated separately by the user, and can be located in a memory space different from the local one (for
example, a SoA located in a GPU device memory can be fully pre-defined on the host and the resulting structure is
passed to the GPU kernel).

This columnar storage allows efficient memory access by GPU kernels (coalesced access on cache line aligned data)
and possibly vectorization.

Additionally, templation of the layout and view classes allows compile-time variations of accesses and checks:
verification of alignment and corresponding compiler hinting, cache strategy (non-coherent, streaming with immediate
invalidation), range checking.

Macro generation allows generating code that provides a clear and concise access of data when used. The code
generation uses the Boost Preprocessing library.

## Layout

`SoALayout` is a macro generated templated class that subdivides a provided buffer into a collection of columns,
Eigen columns and scalars. The buffer is expected to be aligned with a selectable alignment defaulting to the CUDA
GPU cache line (128 bytes). All columns and scalars within a `SoALayout` will be individually aligned, leaving
padding at the end of each if necessary. Eigen columns have each component of the vector or matrix properly aligned
in individual column (by defining the stride between components). Only compile-time sized Eigen vectors and matrices
are supported. Scalar members are members of layout with one element, irrespective of the size of the layout.

Static utility functions automatically compute the byte size of a layout, taking into account all its columns and
alignment.

## View

Layout classes also define a `View` and `ConstView` subclass that provide access to each column and
scalar of the layout. In addition to those fully parametrized templates, two others levels of parametrization are
provided: `ViewTemplate`, `ViewViewTemplateFreeParams` and respectively `ConstViewTemplate`,
`ConstViewTemplateFreeParams`. The parametrization of those templates is explained in the [Template
parameters section](#template-parameters).

The view can be generated in a constant and non-constant flavors. All view flavors provide with the same
interface where scalar elements are accessed with an `operator()`: `soa.scalar()` while columns (Eigen or not) are
accessed via a array of structure (AoS) -like syntax: `soa[index].x()`. The "struct" object returned by `operator[]`
can be used as a shortcut: `auto si = soa[index]; si.z() = si.x() + si.y();`

A view can be instanciated by being passed the corresponding layout or passing from the [Metarecords subclass](#metarecords-subclass).
This view can point to data belonging to different SoAs and thus not contiguous in memory.

## Descriptor

The nested class `ConstDescriptor` can only be instantiated passing a `View` or a `ConstView` and provides access to columns
and related information. This class should be considered an internal implementation detail,
used solely by the SoA and EDM frameworks for performing heterogeneous memory operations. It is used to implement the
`deepCopy` from a `View` referencing different memory buffers, as shown in 
[`PortableHostCollection<T>`](../../DataFormats/Portable/README.md#portablehostCollection)
and [`PortableDeviceCollection<T, TDev>`](../../DataFormats/Portable/README.md#portabledeviceCollection) sections.
More specifically, it provides access to the each column through a `std::tuple<std::span<T>...>` accessible via `descriptor.buff`
as well as the types of the columns via a `std::tuple<cms::soa::SoAColumnType>` accessible via `descriptor.columnTypes`.

## Metadata subclass

In order to no clutter the namespace of the generated class, a subclass name `Metadata` is generated. It is
instanciated with the `metadata()` member function and contains various utility functions, like `size()` (number
of elements in the SoA), `byteSize()`, `byteAlignment()`, `data()` (a pointer to the buffer). A `nextByte()`
function computes the first byte of a structure right after a layout, allowing using a single buffer for multiple
layouts.

## Metarecords subclass

The nested type `Metarecords` describes the elements of the SoA. It can be instantiated by the `records()` member 
function of a `View` or `ConstView`. Every object contains the address of the first element of the column, the number
of elements per column, and the stride for the Eigen columns. These are used to validate the columns size at run time 
and to build a generic `View` as described in [View](#view).

## Customized methods

It is possible to generate methods inside the `element` and `const_element` nested structs using the `SOA_ELEMENT_METHODS`
and `SOA_CONST_ELEMENT_METHODS` macros. Each of these macros can be called only once, and can define multiple methods. All the methods need to be declared as `constexpr` to work in a heterogenous environment and to avoid the dependency on alpaka for the data formats.
[An example is showed below.](#examples)

## Blocks

`SoABlocks` is a macro-generated templated class that enables structured composition of multiple `SoALayouts` into a single
container, referred to as "blocks". Each block is a Layout, and the structure itself looks like multiple contigous memory 
buffers of different sizes. The alignment is ensured to be the same for every block. `SoABlocks` also supports 
`View` and `ConstView` classes. In addition to those fully parametrized templates, two further levels of parametrization are provided:
`ViewTemplate`, `ViewTemplateFreeParams` and respectively `ConstViewTemplate`, `ConstViewTemplateFreeParams`, 
mirroring the structure of the underlying structs. The blocks are built via composition and access to individual layouts 
and views is provided by name.

TODOs:
- Add introspection utilities to print the structure and layout of a `SoABlocks` instance.
- Implement support for heterogeneous `deepCopy()` operations between different but compatible `SoABlocks` configurations.

[An example of utilization is showed below.](#examples)

## ROOT serialization and de-serialization

Layouts can be serialized and de-serialized with ROOT. In order to generate the ROOT dictionary, separate
`clases_def.xml` and `classes.h` should be prepared. `classes.h` ensures the inclusion of the proper header files to
get the definition of the serialized classes, and `classes_def.xml` needs to define the fixed list of members that
ROOT should ignore, plus the list of all the columns. [An example is provided below.](#examples)

Serialization of Eigen data is not yet supported.

## Template parameters

The template shared by layouts and parameters are:
- Byte aligment (defaulting to the nVidia GPU cache line size (128 bytes))
- Alignment enforcement (`relaxed` or `enforced`). When enforced, the alignment will be checked at construction
  time.~~, and the accesses are done with compiler hinting (using the widely supported `__builtin_assume_aligned`
  intrinsic).~~ It turned out that hinting `nvcc` for alignement removed the benefit of more important `__restrict__`
  hinting. The `__builtin_assume_aligned` is hence currently not use.

In addition, the views also provide access parameters:
- Restrict qualify: add restrict hints to read accesses, so that the compiler knows it can relax accesses to the
  data and assume it will not change. On nVidia GPUs, this leads to the generation of instruction using the faster
  non-coherent cache.
- Range checking: add index checking on each access. As this is a compile time parameter, the cost of the feature at
  run time is null if turned off. When turned on, the accesses will be slowed down by checks. Uppon error detection,
  an exception is launched (on the CPU side) or the kernel is made to crash (on the GPU side). This feature can help
  the debugging of index issues at runtime, but of course requires a recompilation.

The trivial views subclasses come in a variety of parametrization levels: `View` uses the same byte
alignement and alignment enforcement as the layout, and defaults (off) for restrict qualifying and range checking.
`ViewTemplate` template allows setting of restrict qualifying and range checking, while
`ViewTemplateFreeParams` allows full re-customization of the template parameters.

## Using SoA layouts and views with GPUs

Instanciation of views and layouts is preferably done on the CPU side. The view object is lightweight, with only one
pointer per column, plus the global number of elements. Extra view class can be generated to restrict this number of
pointers to the strict minimum in scenarios where only a subset of columns are used in a given GPU kernel.

## Examples

A layout can be defined as:

```C++
#include "DataFormats/SoALayout.h"

GENERATE_SOA_LAYOUT(SoA1LayoutTemplate,
  // predefined static scalars
  // size_t size;
  // size_t alignment;

  // columns: one value per element
  SOA_COLUMN(double, x),
  SOA_COLUMN(double, y),
  SOA_COLUMN(double, z),
  SOA_EIGEN_COLUMN(Eigen::Vector3d, a),
  SOA_EIGEN_COLUMN(Eigen::Vector3d, b),
  SOA_EIGEN_COLUMN(Eigen::Vector3d, r),
  SOA_COLUMN(uint16_t, color),
  SOA_COLUMN(int32_t, value),
  SOA_COLUMN(double *, py),
  SOA_COLUMN(uint32_t, count),
  SOA_COLUMN(uint32_t, anotherCount),

  // scalars: one value for the whole structure
  SOA_SCALAR(const char *, description),
  SOA_SCALAR(uint32_t, someNumber)
);

// Default template parameters are <
//   size_t ALIGNMENT = cms::soa::CacheLineSize::defaultSize,
//   bool ALIGNMENT_ENFORCEMENT = cms::soa::AlignmentEnforcement::relaxed
// >
using SoA1Layout = SoA1LayoutTemplate<>;

using SoA1LayoutAligned = SoA1LayoutTemplate<cms::soa::CacheLineSize::defaultSize, cms::soa::AlignmentEnforcement::enforced>;
```

It is possible to declare `constexpr` methods that operate on the SoA elements:

```C++
#include "DataFormats/SoALayout.h"

GENERATE_SOA_LAYOUT(SoATemplate,
  SOA_COLUMN(double, x),
  SOA_COLUMN(double, y),
  SOA_COLUMN(double, z),
  
  // methods operating on const_element
  SOA_CONST_ELEMENT_METHODS(
    constexpr auto norm() const {
      return sqrt(x()*x() + y()+y() + z()*z());
    }
  ),

  // methods operating on element
  SOA_ELEMENT_METHODS(
    constexpr void scale(float arg) {
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

The buffer of the proper size is allocated, and the layout is populated with:

```C++
// Allocation of aligned
size_t elements = 100;
using AlignedBuffer = std::unique_ptr<std::byte, decltype(std::free) *>;
AlignedBuffer h_buf (reinterpret_cast<std::byte*>(aligned_alloc(SoA1LayoutAligned::alignment, SoA1LayoutAligned::computeDataSize(elements))), std::free);
SoA1LayoutAligned soaLayout(h_buf.get(), elements);
```

The mutable and const views with the exact same set of columns and their parametrized variants are provided from the layout as:

```C++
// (Pseudo-code)
struct SoA1Layout::View;

template<bool RESTRICT_QUALIFY = cms::soa::RestrictQualify::enabled,
         bool RANGE_CHECKING = cms::soa::RangeChecking::disabled>
struct SoA1Layout::ViewTemplate;

template<size_t ALIGNMENT = cms::soa::CacheLineSize::defaultSize,
         bool ALIGNMENT_ENFORCEMENT = cms::soa::AlignmentEnforcement::relaxed,
         bool RESTRICT_QUALIFY = cms::soa::RestrictQualify::enabled,
         bool RANGE_CHECKING = cms::soa::RangeChecking::disabled>
struct SoA1Layout::ViewTemplateFreeParams;

struct SoA1Layout::ConstView;

template<bool RESTRICT_QUALIFY = cms::soa::RestrictQualify::enabled,
         bool RANGE_CHECKING = cms::soa::RangeChecking::disabled>
struct SoA1Layout::ConstViewTemplate;

template<size_t ALIGNMENT = cms::soa::CacheLineSize::defaultSize,
         bool ALIGNMENT_ENFORCEMENT = cms::soa::AlignmentEnforcement::relaxed,
         bool RESTRICT_QUALIFY = cms::soa::RestrictQualify::enabled,
         bool RANGE_CHECKING = cms::soa::RangeChecking::disabled>
struct SoA1Layout::ConstViewTemplateFreeParams;
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

```
                   
## Current status and further improvements

### Available features

- The layout and views support scalars and columns, alignment and alignment enforcement and hinting (linked).
- Automatic `__restrict__` compiler hinting is supported and can be enabled where appropriate.
- Automatic creation of trivial views and const views derived from a single layout.
- Cache access style, which was explored, was abandoned as this not-yet-used feature interferes with `__restrict__`
  support (which is already in used in existing code). It could be made available as a separate tool that can be used
  directly by the module developer, orthogonally from SoA.
- Optional (compile time) range checking validates the index of every column access, throwing an exception on the
  CPU side and forcing a segmentation fault to halt kernels. When not enabled, it has no impact on performance (code
  not compiled)
- Eigen columns are also suported, with both const and non-const flavors.
- ROOT serialization and deserialization is supported. In CMSSW, it is planned to be used through the memory
  managing `PortableCollection` family of classes.
- An `operator<<()` is provided to print the layout of an SoA to standard streams.
