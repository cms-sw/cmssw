# Structure of array (SoA) generation

The two header files [`SoALayout.h`](SoALayout.h) and [`SoAView.h`](SoAView.h) define preprocessor macros that allow generating SoA 
classes. The SoA classes generate multiple, aligned column from a memory buffer. The memory buffer is allocated separately by the
user, and can be located in a memory space different from the local one (for example, a SoA located in a GPU device memory is be
fully pre-defined on the host and the resulting structure is passed to the GPU kernel).

This columnar storage allows efficient memory access by GPU kernels (coalesced access on cache line aligned data) and possibly 
vectorization.

Additionally, templation of the layout and view classes will allow compile-time variations of accesses and checks: verification of 
alignment and corresponding compiler hinting, cache strategy (non-coherent, streaming with immediate invalidation), range checking.

## Layout

`SoALayout` is a macro generated templated class that subdivides a provided buffer into a collection of columns, Eigen columns and 
scalars. The buffer is expected to be aligned with a selectable alignment defaulting to the CUDA GPU cache line (128 bytes). All 
columns and scalars within a `SoALayout` will be individually aligned, leaving padding at the end of each if necessary. Eigen columns 
have each component of the vector or matrix properly aligned in individual column (by defining the stride between components). Only 
compile-time sized Eigen vectors and matrices are supported. Scalar members are members of layout with one element, irrespective of 
the size of the layout.

Static utility functions automatically compute the byte size of a layout, taking into account all its columns and alignment.

## View

`SoAView` is a macro generated templated class allowing access to columns defined in one or multiple `SoALayout`s or `SoAViews`. The 
view can be generated in a constant and non-constant flavors. All view flavors provide with the same interface where scalar elements 
are accessed with an `operator()`: `soa.scalar()` while columns (Eigen or not) are accessed via a array of structure (AoS) -like 
syntax: `soa[index].x()`. The "struct" object returned by `operator[]` can be used as a shortcut: 
`auto si = soa[index]; si.z() = si.x() + zi.y();`

A view can be instanciated by being passed the layout(s) and view(s) it is defined against, or column by column.

## SoAMetadata subclass

In order to no clutter the namespace of the generated class, a subclass name `SoAMetadata` is generated. Its instanciated with the 
`soaMetadata()` member function and contains various utility functions, like `size()` (number of elements in the SoA), `byteSize()`, 
`byteAlignment()`, `data()` (a pointer to the buffer). A `nextByte()` function computes the first byte of a structure right after a 
layout, allowing using a single buffer for multiple layouts.

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
//   cms::soa::AlignmentEnforcement ALIGNMENT_ENFORCEMENT = cms::soa::AlignmentEnforcement::Relaxed
// >
using SoA1Layout = SoA1LayoutTemplate<>;

using SoA1LayoutAligned = SoA1LayoutTemplate<cms::soa::CacheLineSize::defaultSize, cms::soa::AlignmentEnforcement::Enforced>;
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

for (size_t i=0; i < soaLayout.soaMetadata().size(); ++i) {
  auto si = soaView[i];
  si.x() = si.y() = i;
  soaView.someNumber() += i;
}
```
Any mixture of mutable and const views can also be defined automatically with the layout (for the trivially identical views) using one those macros `GENERATE_SOA_LAYOUT_VIEW_AND_CONST_VIEW`, `GENERATE_SOA_LAYOUT_AND_VIEW` and `GENERATE_SOA_LAYOUT_AND_CONST_VIEW`:

```C++
GENERATE_SOA_LAYOUT_VIEW_AND_CONST_VIEW(SoA1LayoutTemplate, SoA1ViewTemplate, SoA1ConstViewTemplate,
  // columns: one value per element
  SOA_COLUMN(double, x),
  SOA_COLUMN(double, y),
  SOA_COLUMN(double, z),
  SOA_COLUMN(double, sum),
  SOA_COLUMN(double, prod),
  SOA_COLUMN(uint16_t, color),
  SOA_COLUMN(int32_t, value),
  SOA_COLUMN(double *, py),
  SOA_COLUMN(uint32_t, count),
  SOA_COLUMN(uint32_t, anotherCount),

  // scalars: one value for the whole structure
  SOA_SCALAR(const char *, description),
  SOA_SCALAR(uint32_t, someNumber)
)
```

## Template parameters

The template parameters are:
- Byte aligment (defaulting to the nVidia GPU cache line size (128 bytes))
- Alignment enforcement (`Relaxed` or `Enforced`). When enforced, the alignment will be checked at construction time, and the accesses 
are done with compiler hinting (using the widely supported `__builtin_assume_aligned` intrinsic).

## Using SoA layouts and views with GPUs

Instanciation of views and layouts is preferably done on the CPU side. The view object is lightweight, with only one pointer per 
column (size to be added later). Extra view class can be generated to restrict this number of pointers to the strict minimum in 
scenarios where only a subset of columns are used in a given GPU kernel.

## Current status and further improvements

### Available features

- The layout and views support scalars and columns, alignment and alignment enforcement and hinting.
- Automatic `__restrict__` compiler hinting is supported.
- A shortcut alloCreate a mechanism to derive trivial views and const views from a single layout.
- Cache access style, which was explored, was abandoned as this not-yet-used feature interferes with `__restrict__` support (which is
already in used in existing code). It could be made available as a separate tool that can be used directly by the module developer,
orthogonally from SoA.
- Optional (compile time) range checking validates the index of every column access, throwing an exception on the CPU side and forcing
a segmentation fault to halt kernels. When not enabled, it has no impact on performance (code not compiled)
- Eigen columns are also suported, with both const and non-const flavors.

### Planned additions
- Improve `dump()` function and turn it into a more classic `operator<<()`.
