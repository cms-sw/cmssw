#include "DataFormats/Common/interface/View.h"
#include <typeinfo>

namespace edm {
  //------------------------------------------------------------------
  // Implementation of ViewBase.
  //------------------------------------------------------------------

  ViewBase::~ViewBase() {}

  std::unique_ptr<ViewBase> ViewBase::clone() const {
    auto p = doClone();
#if !defined(NDEBUG)
    //move side-effect out of typeid to avoid compiler warning
    auto p_get = p.get();
    assert(typeid(*p_get) == typeid(*this) && "doClone() incorrectly overriden");
#endif
    return p;
  }

  ViewBase::ViewBase() {}

  ViewBase::ViewBase(ViewBase const&) {}

}  // namespace edm
