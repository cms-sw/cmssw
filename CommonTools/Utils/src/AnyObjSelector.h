#ifndef CommonTools_Utils_AnyObjSelector_h
#define CommonTools_Utils_AnyObjSelector_h
#include "CommonTools/Utils/interface/parser/SelectorBase.h"

namespace reco {
  namespace parser {
    class AnyObjSelector : public SelectorBase {
      bool operator()(const edm::ObjectWithDict& c) const override { return true; }
    };
  }  // namespace parser
}  // namespace reco

#endif
