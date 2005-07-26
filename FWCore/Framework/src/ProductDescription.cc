#include "FWCore/Framework/interface/ProductDescription.h"

/*----------------------------------------------------------------------

$Id: ProductDescription.cc,v 1.1 2005/07/21 13:21:26 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  ProductDescription::ProductDescription() :
    module(),
    product_id(),
    full_product_type_name(),
    friendly_product_type_name(),
    product_instance_name(),
    branchKey()
  { }

  ProductDescription::ProductDescription(ModuleDescription const& m,
      std::string const& name, std::string const& fName, std::string const& pin) :
    module(m),
    product_id(),
    full_product_type_name(name),
    friendly_product_type_name(fName),
    product_instance_name(pin),
    branchKey(fName, m.module_label, pin, m.process_name)
  {}

  void
  ProductDescription::init() const {
       branchKey = BranchKey(friendly_product_type_name, module.module_label, product_instance_name, module.process_name); 
  }

  void
  ProductDescription::write(std::ostream& ) const {
    // To be filled in later.
  }

}

