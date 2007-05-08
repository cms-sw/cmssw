/*----------------------------------------------------------------------
  
$Id: EDProduct.cc,v 1.3 2007/01/11 23:39:18 paterno Exp $

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/EDProduct.h"

namespace edm {
  EDProduct::EDProduct() {}

  EDProduct::~EDProduct() {}

  void EDProduct::fillView(std::vector<void const*>& pointers,
			   std::vector<helper_ptr>& helpers) const
  {
    assert(pointers.empty() && helpers.empty());

    do_fillView(pointers, helpers);
  }
}
