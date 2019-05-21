#include "DataFormats/Common/interface/ValidHandle.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace edm::vhhelper {
  void throwIfNotValid(const void* iProduct) noexcept(false) {
    if (nullptr == iProduct) {
      throw cms::Exception("Invalid Product") << "Attempted to fill a edm::ValidHandle with an invalid product";
    }
  }
}  // namespace edm::vhhelper
