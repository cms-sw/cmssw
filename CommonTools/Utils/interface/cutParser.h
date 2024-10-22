#ifndef CommonTools_Utils_cutParset_h
#define CommonTools_Utils_cutParset_h
#include "CommonTools/Utils/interface/parser/SelectorPtr.h"
#include "CommonTools/Utils/interface/parser/Exception.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"
#include <string>

namespace reco {
  namespace parser {
    bool cutParser(const edm::TypeWithDict &t, const std::string &cut, SelectorPtr &sel, bool lazy);

    template <typename T>
    inline bool cutParser(const std::string &cut, SelectorPtr &sel, bool lazy = false) {
      return reco::parser::cutParser(edm::TypeWithDict(typeid(T)), cut, sel, lazy);
    }
  }  // namespace parser
}  // namespace reco

#endif
