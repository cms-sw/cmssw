#ifndef DataFormats_Portable_interface_PortableCollectionCommon_h
#define DataFormats_Portable_interface_PortableCollectionCommon_h

#include <format>
#include <limits>
#include <stdexcept>
#include <typeinfo>

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/TypeDemangler.h"

namespace portablecollection {

  template <int I, typename TQueue, typename Descriptor, typename ConstDescriptor>
  void deepCopy(TQueue& queue, Descriptor& dest, ConstDescriptor const& src) {
    if constexpr (I < ConstDescriptor::num_cols) {
      assert(std::get<I>(dest.buff).size_bytes() == std::get<I>(src.buff).size_bytes());
      alpaka::memcpy(
          queue,
          alpaka::createView(alpaka::getDev(queue), std::get<I>(dest.buff).data(), std::get<I>(dest.buff).size()),
          alpaka::createView(alpaka::getDev(queue), std::get<I>(src.buff).data(), std::get<I>(src.buff).size()));
      deepCopy<I + 1>(queue, dest, src);
    }
  }

  template <std::integral Int>
  constexpr int size_cast(Int input) {
    if ((std::is_signed_v<Int> && input < 0) || input > std::numeric_limits<int>::max()) {
      throw std::runtime_error(
          std::format("Invalid input value for size of PortableCollection: cannot be narrowed to positive int32. "
                      "Source type: {}, value: {} ",
                      edm::typeDemangle(typeid(Int).name()),
                      input));
    }
    return static_cast<int>(input);
  }

}  // namespace portablecollection

#endif  // DataFormats_Portable_interface_PortableCollectionCommon_h
