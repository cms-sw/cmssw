#ifndef PhysicsTools_Utilities_AnyObjSelector_h
#define PhysicsTools_Utilities_AnyObjSelector_h
#include "PhysicsTools/Utilities/src/SelectorBase.h"

namespace reco {
  namespace parser {
    class AnyObjSelector : public SelectorBase {
      virtual bool operator()(const ROOT::Reflex::Object & c) const { return true; }
    };
  }
}

#endif
