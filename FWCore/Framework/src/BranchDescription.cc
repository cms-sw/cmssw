#include "FWCore/Framework/interface/BranchDescription.h"

/*----------------------------------------------------------------------

$Id: BranchDescription.cc,v 1.10 2005/09/28 04:53:20 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  BranchDescription::BranchDescription() :
    module(),
    productID_(),
    fullClassName_(),
    friendlyClassName_(),
    productInstanceName_(),
    productPtr_(0)
  { }

  BranchDescription::BranchDescription(ModuleDescription const& md,
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
  BranchDescription::init() const {
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
  BranchDescription::write(std::ostream& os) const {
    os << module << std::endl;
    os << "Product ID = " << productID_ << '\n';
    os << "Class Name = " << fullClassName_ << '\n';
    os << "Friendly Class Name = " << friendlyClassName_ << '\n';
    os << "Product Instance Name = " << productInstanceName_ << std::endl;
  }

  bool
  BranchDescription::operator<(BranchDescription const& rh) const {
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
  BranchDescription::operator==(BranchDescription const& rh) const {
    return !((*this) < rh || rh < (*this));
  }
}
