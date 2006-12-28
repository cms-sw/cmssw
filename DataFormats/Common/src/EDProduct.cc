/*----------------------------------------------------------------------
  
$Id: EDProduct.cc,v 1.1 2006/02/07 07:01:51 wmtan Exp $

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/EDProduct.h"

namespace edm {
  EDProduct::EDProduct() {}

  EDProduct::~EDProduct() {}

  void EDProduct::fillView(std::vector<void*>& pointers) const
  {
    do_fillView(pointers);
  }

  void EDProduct::do_fillView(std::vector<void*>&) const { }
}
