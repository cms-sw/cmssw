#include "DataFormats/Portable/interface/PortableCollectionCommon.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace portablecollection {

  [[noreturn]] void throwException(const std::string& category, const std::string& message) {
    throw cms::Exception(category) << message;
  }

}  // namespace portablecollection
