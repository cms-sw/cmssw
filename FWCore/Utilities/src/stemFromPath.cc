#include "FWCore/Utilities/interface/stemFromPath.h"

namespace edm {
  std::string_view stemFromPath(const std::string& path) {
    auto begin = path.rfind("/");
    if (begin == std::string::npos) {
      begin = path.rfind(":");
      if (begin == std::string::npos) {
        // shouldn't really happen?
        begin = 0;
      } else {
        begin += 1;
      }
    } else {
      begin += 1;
    }
    auto end = path.find(".", begin);
    return std::string_view(path).substr(begin, end - begin);
  }
}  // namespace edm
