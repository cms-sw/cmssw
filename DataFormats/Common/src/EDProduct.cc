/*----------------------------------------------------------------------
  
$Id: EDProduct.cc,v 1.7 2007/08/06 22:16:50 wmtan Exp $

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Provenance/interface/ProductID.h"

namespace edm {
  EDProduct::EDProduct() {}

  EDProduct::~EDProduct() {}

  void EDProduct::fillView(ProductID const& id,
			   std::vector<void const*>& pointers,
			   helper_vector_ptr& helpers) const
  {
    // This should never be called with non-empty arguments, or an
    // invalid ID; any attempt to do so is an indication of a coding
    // error.
    assert(id.isValid());
    assert(pointers.empty());
    assert(helpers.get() == 0);

    do_fillView(id, pointers, helpers);
  }
  
  void EDProduct::setPtr(const std::type_info& iToType,
                         unsigned long iIndex,
                         void const*& oPtr) const
{
  do_setPtr(iToType, iIndex, oPtr);
}
  
}
