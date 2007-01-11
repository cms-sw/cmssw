/*----------------------------------------------------------------------
  
$Id: EDProduct.cc,v 1.2 2006/12/28 18:51:12 paterno Exp $

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/EDProduct.h"

namespace edm {
  EDProduct::EDProduct() {}

  EDProduct::~EDProduct() {}

  void EDProduct::fillView(std::vector<void const*>& pointers) const
  {
    assert(pointers.empty());
    do_fillView(pointers);
  }
}
