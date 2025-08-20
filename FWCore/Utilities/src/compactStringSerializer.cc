#include "FWCore/Utilities/interface/compactStringSerializer.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace edm::compactString::detail {
  void throwIfContainsDelimiters(std::string const& str) {
    auto pos = str.find_first_of(kDelimiters);
    if (pos != std::string::npos) {
      cms::Exception ex("compactString");
      ex << "Serialized string '" << str << "' contains ";
      if (str[pos] == kContainerDelimiter) {
        ex << "container";
      } else {
        ex << "element";
      }
      ex << " delimiter at position " << pos;
      throw ex;
    }
  }
}  // namespace edm::compactString::detail
