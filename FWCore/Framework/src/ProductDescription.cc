#include "FWCore/Framework/interface/ProductDescription.h"

/*----------------------------------------------------------------------

$Id: ProductDescription.cc,v 1.4 2005/07/14 22:50:53 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  ProductDescription::ProductDescription() :
    module(),
    product_id(),
    full_product_type_name(),
    friendly_product_type_name(),
    product_instance_name()
  { }

  ProductDescription::ProductDescription(const ModuleDescription& m) :
    module(m),
    product_id(),
    full_product_type_name(),
    friendly_product_type_name(),
    product_instance_name()
  { }

  void
  ProductDescription::write(std::ostream& ) const {
    // To be filled in later.
  }

}

