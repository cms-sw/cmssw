#ifndef PhysicsTools_NanoAOD_FlatTableColumnDispatch_h
#define PhysicsTools_NanoAOD_FlatTableColumnDispatch_h

#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <cstdint>
#include <type_traits>

// Calls func with a std::type_identity tag naming the C++ type that backs the given FlatTable
// column type, so that code handling all ten column types can be written once:
//
//   dispatchColumnType(table.columnType(i), [&](auto tag) {
//     using ColumnT = typename decltype(tag)::type;
//     ...
//   });
//
// Note that the storage type of a column is nanoaod::FlatTable::ColumnStorageType<ColumnT>, which
// differs from ColumnT for bool: table.columnData<bool>() hands back a span of std::uint8_t.
//
// The switch deliberately has no default case. -Werror=switch then turns a column type added to
// nanoaod::FlatTable into a compile error here, rather than something silently unwritten.
template <typename F>
decltype(auto) dispatchColumnType(nanoaod::FlatTable::ColumnType type, F&& func) {
  using ColumnType = nanoaod::FlatTable::ColumnType;
  switch (type) {
    case ColumnType::UInt8:
      return func(std::type_identity<std::uint8_t>{});
    case ColumnType::Int16:
      return func(std::type_identity<std::int16_t>{});
    case ColumnType::UInt16:
      return func(std::type_identity<std::uint16_t>{});
    case ColumnType::Int32:
      return func(std::type_identity<std::int32_t>{});
    case ColumnType::UInt32:
      return func(std::type_identity<std::uint32_t>{});
    case ColumnType::Int64:
      return func(std::type_identity<std::int64_t>{});
    case ColumnType::UInt64:
      return func(std::type_identity<std::uint64_t>{});
    case ColumnType::Bool:
      return func(std::type_identity<bool>{});
    case ColumnType::Float:
      return func(std::type_identity<float>{});
    case ColumnType::Double:
      return func(std::type_identity<double>{});
  }
  throw cms::Exception("UnsupportedType")
      << "Unknown nanoaod::FlatTable::ColumnType " << static_cast<int>(type) << "\n";
}

#endif
