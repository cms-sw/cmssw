#include "FWCore/Framework/interface/ESValidHandle.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace edm::esvhhelper {
  void
  throwIfNotValid(const void* iProduct) noexcept(false) {
    if(nullptr == iProduct) {
      throw cms::Exception("Invalid Product")<<"Attempted to fill a edm::ESValidHandle with an invalid product";
    }
  }
}
