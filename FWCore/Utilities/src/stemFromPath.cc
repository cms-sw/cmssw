#include "FWCore/Utilities/interface/stemFromPath.h"

namespace edm {
  std::string_view stemFromPath(std::string_view path) {
    auto begin = path.rfind('/');
    if (begin == std::string_view::npos) {
      begin = path.rfind(':');
      if (begin == std::string_view::npos) {
        // shouldn't really happen?
        begin = 0;
      } else {
        begin += 1;
      }
    } else {
      begin += 1;
    }
    auto end = path.find('.', begin);
    return path.substr(begin, end - begin);
  }
}  // namespace edm
