#ifndef DataFormats_Portable_interface_PortableCollectionCommon_h
#define DataFormats_Portable_interface_PortableCollectionCommon_h

#include <format>
#include <limits>
#include <stdexcept>
#include <typeinfo>

#include "FWCore/Utilities/interface/TypeDemangler.h"

namespace portablecollection {

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
