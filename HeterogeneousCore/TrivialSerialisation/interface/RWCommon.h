#ifndef HeterogeneousCore_TrivialSerialisation_interface_RWCommon_h
#define HeterogeneousCore_TrivialSerialisation_interface_RWCommon_h

#include <cstddef>
#include <span>
#include <vector>

#include "DataFormats/TrivialSerialisation/interface/MemoryCopyTraits.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/AnyBuffer.h"

namespace ngt {

  // Helper functions used by both the alpaka and non-alpaka variants of the
  // Reader and Writer.

  // Return the TrivialCopyProperties for an object, wrapped in an AnyBuffer.
  // Returns an empty AnyBuffer if MemoryCopyTraits<T> has no properties().
  template <typename T>
  AnyBuffer get_properties(T const& obj) {
    if constexpr (not HasTrivialCopyProperties<T>) {
      return {};
    } else {
      return AnyBuffer(MemoryCopyTraits<T>::properties(obj));
    }
  }

  // Return the memory regions of an object (mutable)
  template <typename T>
  std::vector<std::span<std::byte>> get_regions(T& obj) {
    static_assert(HasRegions<T>);
    return MemoryCopyTraits<T>::regions(obj);
  }

  // Return the memory regions of an object (const)
  template <typename T>
  std::vector<std::span<const std::byte>> get_regions(T const& obj) {
    static_assert(HasRegions<T>);
    return MemoryCopyTraits<T>::regions(obj);
  }

  // Call finalize() on an object, if MemoryCopyTraits<T> provides it
  template <typename T>
  void do_finalize(T& obj) {
    if constexpr (HasTrivialCopyFinalize<T>) {
      MemoryCopyTraits<T>::finalize(obj);
    }
  }

}  // namespace ngt

#endif  // HeterogeneousCore_TrivialSerialisation_interface_RWCommon_h
