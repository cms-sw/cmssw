#include "DataFormats/Common/interface/View.h"
#include <typeinfo>

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
    assert(typeid(*(p.get()))==typeid(*this) && "doClone() incorrectly overriden");
    return p;
  }

  ViewBase::ViewBase() {}

  ViewBase::ViewBase(ViewBase const&) { }

}
