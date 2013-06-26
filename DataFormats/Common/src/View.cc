#include "DataFormats/Common/interface/View.h"

namespace edm
{
  //------------------------------------------------------------------
  // Implementation of ViewBase.
  //------------------------------------------------------------------


  ViewBase::~ViewBase() { }

  ViewBase*
  ViewBase::clone() const
  {
    ViewBase* p = doClone();
    assert(typeid(*p)==typeid(*this) && "doClone() incorrectly overriden");
    return p;
  }

  ViewBase::ViewBase() {}

  ViewBase::ViewBase(ViewBase const&) { }

}
