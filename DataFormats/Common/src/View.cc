#include "DataFormats/Common/interface/View.h"

namespace edm
{
  //------------------------------------------------------------------
  // Implementation of ViewBase.
  //------------------------------------------------------------------


  ViewBase::~ViewBase() { }

  std::unique_ptr<ViewBase>
  ViewBase::clone() const
  {
    auto p = doClone();
    assert(typeid(*p)==typeid(*this) && "doClone() incorrectly overriden");
    return p;
  }

  ViewBase::ViewBase() {}

  ViewBase::ViewBase(ViewBase const&) { }

}
