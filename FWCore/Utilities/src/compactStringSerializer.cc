#include "FWCore/Utilities/interface/compactStringSerializer.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace edm::compactString::detail {
  void throwIfContainsDelimiters(std::string const& str) {
    auto pos = str.find_first_of(kDelimiters);
    if (pos != std::string::npos) {
      throw cms::Exception("compactString") << "Serialized string '" << str << "' contains a delimiter";
    }
  }
}  // namespace edm::compactString::detail
