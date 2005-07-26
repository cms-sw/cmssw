#include "FWCore/Framework/interface/ProductDescription.h"

/*----------------------------------------------------------------------

$Id: ProductDescription.cc,v 1.2 2005/07/26 04:42:28 wmtan Exp $

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
    branchKey() {
      init();
    }

  void
  ProductDescription::init() const {
    branchKey = BranchKey(friendly_product_type_name, module.module_label,
       product_instance_name, module.process_name); 

    char const underscore('_');
    char const period('.');
    std::string const prod("PROD");

    if (module.process_name == prod) {
      if (product_instance_name.empty()) {
        branchName = friendly_product_type_name + underscore + module.module_label + period;
        return;
      }
      branchName = friendly_product_type_name + underscore + module.module_label + underscore +
        product_instance_name + period;
      return;
    }
    branchName = friendly_product_type_name + underscore + module.module_label + underscore +
      product_instance_name + underscore + module.process_name + period;
  }

  void
  ProductDescription::write(std::ostream& ) const {
    // To be filled in later.
  }

}

