#include "DataFormats/Common/interface/BranchDescription.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Utilities/interface/Exception.h"

/*----------------------------------------------------------------------

$Id: BranchDescription.cc,v 1.6 2006/06/24 01:47:34 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  BranchDescription::BranchDescription() :
    module(),
    productID_(),
    fullClassName_(),
    friendlyClassName_(),
    productInstanceName_(),
    branchAlias_()
  { }

  BranchDescription::BranchDescription(ModuleDescription const& md,
				       std::string const& name, 
				       std::string const& fName, 
				       std::string const& pin, 
				       std::string const& alias) :
    module(md),
    productID_(),
    fullClassName_(name),
    friendlyClassName_(fName),
    productInstanceName_(pin),
    branchAlias_(alias) {
    init();
  }

  void
  BranchDescription::init() const {
    char const underscore('_');
    char const period('.');
    std::string const prod("PROD");

    if (productType().find(underscore) != std::string::npos) {
      throw cms::Exception("IllegalCharacter") << "Class name '" << productType()
      << "' contains an underscore ('_'), which is illegal in the name of a product.\n";
    }

    if (moduleLabel().find(underscore) != std::string::npos) {
      throw cms::Exception("IllegalCharacter") << "Module label '" << moduleLabel()
      << "' contains an underscore ('_'), which is illegal in a module label.\n";
    }

    if (productInstanceName().find(underscore) != std::string::npos) {
      throw cms::Exception("IllegalCharacter") << "Product instance name '" << productInstanceName()
      << "' contains an underscore ('_'), which is illegal in a product instance name.\n";
    }

    if (processName().find(underscore) != std::string::npos) {
      throw cms::Exception("IllegalCharacter") << "Process name '" << processName()
      << "' contains an underscore ('_'), which is illegal in a process name.\n";
    }

    if (processName() == prod) {
      if (productInstanceName().empty()) {
        branchName_ = productType() + underscore + moduleLabel() + period;
        return;
      }
      branchName_ = productType() + underscore + moduleLabel() + underscore +
        productInstanceName() + period;
      return;
    }
    branchName_ = productType() + underscore + moduleLabel() + underscore +
      productInstanceName() + underscore + processName() + period;
  }

  void
  BranchDescription::write(std::ostream& os) const {
    os << module << std::endl;
    os << "Product ID = " << productID() << '\n';
    os << "Class Name = " << className() << '\n';
    os << "Friendly Class Name = " << productType() << '\n';
    os << "Product Instance Name = " << productInstanceName() << std::endl;
  }

  bool
  BranchDescription::operator<(BranchDescription const& rh) const {
    if (productType() < rh.productType()) return true;
    if (rh.productType() < productType()) return false;
    if (productInstanceName() < rh.productInstanceName()) return true;
    if (rh.productInstanceName() < productInstanceName()) return false;
    if (module < rh.module) return true;
    if (rh.module < module) return false;
    if (className() < rh.className()) return true;
    if (rh.className() < className()) return false;
    if (productID() < rh.productID()) return true;
    return false;
  }

  bool
  BranchDescription::operator==(BranchDescription const& rh) const {
    return !((*this) < rh || rh < (*this));
  }
}
