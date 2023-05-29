# Structure of array (SoA) generation

The two header files [`SoALayout.h`](SoALayout.h) and [`SoAView.h`](SoAView.h) define preprocessor macros that
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

`SoAView` is a macro generated templated class allowing access to columns defined in one or multiple `SoALayout`s or
`SoAViews`. The view can be generated in a constant and non-constant flavors. All view flavors provide with the same
interface where scalar elements are accessed with an `operator()`: `soa.scalar()` while columns (Eigen or not) are
accessed via a array of structure (AoS) -like syntax: `soa[index].x()`. The "struct" object returned by `operator[]`
can be used as a shortcut: `auto si = soa[index]; si.z() = si.x() + zi.y();`

A view can be instanciated by being passed the layout(s) and view(s) it is defined against, or column by column.

Layout classes also define a `View` and `ConstView` subclass that provide access to each column and
scalar of the layout. In addition to those fully parametrized templates, two others levels of parametrization are
provided: `ViewTemplate`, `ViewViewTemplateFreeParams` and respectively `ConstViewTemplate`,
`ConstViewTemplateFreeParams`. The parametrization of those templates is explained in the [Template
parameters section](#template-parameters).

## Metadata subclass

In order to no clutter the namespace of the generated class, a subclass name `Metadata` is generated. It is
instanciated with the `metadata()` member function and contains various utility functions, like `size()` (number
of elements in the SoA), `byteSize()`, `byteAlignment()`, `data()` (a pointer to the buffer). A `nextByte()`
function computes the first byte of a structure right after a layout, allowing using a single buffer for multiple
layouts.

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

The buffer of the proper size is allocated, and the layout is populated with:

```C++
// Allocation of aligned
size_t elements = 100;
using AlignedBuffer = std::unique_ptr<std::byte, decltype(std::free) *>;
AlignedBuffer h_buf (reinterpret_cast<std::byte*>(aligned_alloc(SoA1LayoutAligned::byteAlignment, SoA1LayoutAligned::computeDataSize(elements))), std::free);
SoA1LayoutAligned soaLayout(h_buf.get(), elements);
```

A view will derive its column types from one or multiple layouts. The macro generating the view takes a list of layouts or views it
gets is data from as a first parameter, and the selection of the columns the view will give access to as a second parameter.

```C++
// A 1 to 1 view of the layout (except for unsupported types).
GENERATE_SOA_VIEW(SoA1ViewTemplate,
  SOA_VIEW_LAYOUT_LIST(
    SOA_VIEW_LAYOUT(SoA1Layout, soa1)
  ),
  SOA_VIEW_VALUE_LIST(
    SOA_VIEW_VALUE(soa1, x),
    SOA_VIEW_VALUE(soa1, y),
    SOA_VIEW_VALUE(soa1, z),
    SOA_VIEW_VALUE(soa1, color),
    SOA_VIEW_VALUE(soa1, value),
    SOA_VIEW_VALUE(soa1, py),
    SOA_VIEW_VALUE(soa1, count),
    SOA_VIEW_VALUE(soa1, anotherCount),
    SOA_VIEW_VALUE(soa1, description),
    SOA_VIEW_VALUE(soa1, someNumber)
  )
);

using SoA1View = SoA1ViewTemplate<>;

SoA1View soaView(soaLayout);

for (size_t i=0; i < soaLayout.metadata().size(); ++i) {
  auto si = soaView[i];
  si.x() = si.y() = i;
  soaView.someNumber() += i;
}
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
