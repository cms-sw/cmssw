# `ticl::AssociationMap`

`ticl::AssociationMap` is an heterogeneous GPU-friendly associator that maps integral `keys` to values, which mush be `trivially copyable`.
The map is designed using `SoABlocks` so it can be stored in a `PortableCollection` and allocated like a normal SoA.

Internally it uses a CSR-like (compressed sparse row) layout:

- an **offsets** block holding `nkeys + 1` prefix-sum offsets (`keys_offsets`)
- a **content** block holding the `nvalues` values themselves (`values`)

Defined in `interface/AssociationMap.h`.

## Template parameters

```cpp
template <std::integral TKey = uint32_t, ticl::concepts::trivially_copyable TMapped = uint32_t>
```

- `TKey`: an integral type used as the key (must be usable as an index, i.e. `0 <= key < nkeys`).
- `TMapped`: any trivially-copyable type stored as the associated value.

Both `TKey` and `TMapped` default to `uint32_t`.

## Construction

```cpp
#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/TICL/interface/AssociationMap.h"

const uint32_t nkeys = 2;
const uint32_t nvalues = 100;
PortableCollection<Device, ticl::AssociationMap<>> map(queue, nkeys, nvalues);
```

## Filling the map

Filling is done with `ticl::associator::fill`, declared in `interface/FillAssociator.h`.
The fill happens all in one go, so all the data to save inside the map should be available
when calling `fill`.

```cpp
#include "DataFormats/TICL/interface/FillAssociator.h"

// keys[i] and values[i] are the key/value of the i-th (key, value) pair to insert
ticl::associator::fill<Acc1D>(queue, map.view(), keys_span, values_span);
```

- `keys` and `values` are `std::span<const TKey>` / `std::span<const TMapped>` and must have the same size.
- Every entry in `keys` must satisfy `0 <= key < map.keys()`.
- `keys.size() == values.size()` must be `<= nvalues` (the capacity passed at construction).
- `fill` computes, per key, how many values map to it, turns those counts into prefix-sum
  offsets (via a multi-block prefix scan), stores them in the `offsets` block, and scatters each value into its key's slice of the `content` block.
- The relative order of values within a single key's bucket is **not guaranteed** to match the input order, since the scatter happens in parallel across threads.
- `fill` can be called with a shorter span than the map's capacity (a "partial fill"); any keys/values beyond `keys.size()`/`values.size()` are simply not part of the fill.

## Reading the map (host and device)

The accessors are methods of the View/ConstView and are marked `SOA_HOST_DEVICE`, so they can be called both from the host and the device:

```cpp
auto view = map.view();        // or map.const_view() for read-only access

view.keys();                   // number of keys (nkeys)
view.size();                   // total capacity / number of stored values (nvalues)
view.count(key);               // number of values associated to `key`
view.contains(key);            // true if count(key) > 0
view[key];                     // std::span<TMapped> (or std::span<const TMapped> on a
                               // ConstView) with the values associated to `key`
```

Example, inside an alpaka kernel:

```cpp
ALPAKA_FN_ACC void operator()(const Acc1D& acc, ticl::AssociationMapView<> map, ...) const {
  for (auto key : alpaka::uniformElements(acc, map.keys())) {
    if (!map.contains(key))
      continue;
    for (auto value : map[key]) {
      // ...
    }
  }
}
```

### Note
For querying the size of the map prefer the `keys` and `size` view methods rather than the native SoA's `metadata().size()`.

## Relevant files

- `interface/AssociationMap.h` — the SoA layout and the `View`/`ConstView` accessors (`operator[]`, `count`, `contains`, `keys`, `size`).
- `interface/FillAssociator.h` — public entry point `ticl::associator::fill`.
- `interface/detail/FillAssociator.h` — the kernels used to fill the map (counting, prefix-sum, scatter).
- `test/alpaka/TestAssociationMap.dev.cc` — usage examples covering construction, filling (full and partial), deep copy, and access from both host and device code.
