#include "FWCore/Framework/interface/ProductDescription.h"

/*----------------------------------------------------------------------

$Id: ProductDescription.cc,v 1.3 2005/07/26 20:16:21 wmtan Exp $

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

  ProductDescription::ProductDescription(ModuleDescription const& md,
      std::string const& name, std::string const& fName, std::string const& pin) :
    module(md),
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

  bool
  ProductDescription::operator<(ProductDescription const& rh) const {
    if (friendly_product_type_name < rh.friendly_product_type_name) return true;
    if (rh.friendly_product_type_name < friendly_product_type_name) return false;
    if (product_instance_name < rh.product_instance_name) return true;
    if (rh.product_instance_name < product_instance_name) return false;
    if (module < rh.module) return true;
    if (rh.module < module) return false;
    if (full_product_type_name < rh.full_product_type_name) return true;
    if (rh.full_product_type_name < full_product_type_name) return false;
    if (product_id < rh.product_id) return true;
    return false;
  }

  bool
  ProductDescription::operator==(ProductDescription const& rh) const {
    return !((*this) < rh || rh < (*this));
  }
}
