#include "FWCore/Framework/interface/ProductDescription.h"

/*----------------------------------------------------------------------

$Id: ProductDescription.cc,v 1.7 2005/07/30 23:47:52 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  ProductDescription::ProductDescription() :
    module(),
    productID_(),
    fullClassName_(),
    friendlyClassName_(),
    productInstanceName_(),
    productPtr_(0)
  { }

  ProductDescription::ProductDescription(ModuleDescription const& md,
      std::string const& name, std::string const& fName, std::string const& pin, EDProduct const* edp) :
    module(md),
    productID_(),
    fullClassName_(name),
    friendlyClassName_(fName),
    productInstanceName_(pin),
    productPtr_(edp) {
      init();
    }

  void
  ProductDescription::init() const {
    char const underscore('_');
    char const period('.');
    std::string const prod("PROD");

    if (module.processName_ == prod) {
      if (productInstanceName_.empty()) {
        branchName_ = friendlyClassName_ + underscore + module.moduleLabel_ + period;
        return;
      }
      branchName_ = friendlyClassName_ + underscore + module.moduleLabel_ + underscore +
        productInstanceName_ + period;
      return;
    }
    branchName_ = friendlyClassName_ + underscore + module.moduleLabel_ + underscore +
      productInstanceName_ + underscore + module.processName_ + period;
  }

  void
  ProductDescription::write(std::ostream& ) const {
    // To be filled in later.
  }

  bool
  ProductDescription::operator<(ProductDescription const& rh) const {
    if (friendlyClassName_ < rh.friendlyClassName_) return true;
    if (rh.friendlyClassName_ < friendlyClassName_) return false;
    if (productInstanceName_ < rh.productInstanceName_) return true;
    if (rh.productInstanceName_ < productInstanceName_) return false;
    if (module < rh.module) return true;
    if (rh.module < module) return false;
    if (fullClassName_ < rh.fullClassName_) return true;
    if (rh.fullClassName_ < fullClassName_) return false;
    if (productID_ < rh.productID_) return true;
    return false;
  }

  bool
  ProductDescription::operator==(ProductDescription const& rh) const {
    return !((*this) < rh || rh < (*this));
  }
}
