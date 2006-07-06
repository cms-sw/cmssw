#include "DataFormats/Common/interface/BranchDescription.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Utilities/interface/Exception.h"

/*----------------------------------------------------------------------

$Id: BranchDescription.cc,v 1.7.2.3 2006/06/30 04:30:05 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  BranchDescription::BranchDescription() :
    moduleLabel_(),
    processName_(),
    productID_(),
    fullClassName_(),
    friendlyClassName_(),
    productInstanceName_(),
    moduleDescriptionID_(),
    psetIDs_(),
    branchAliases_(),
    produced_(false)
  { }

  BranchDescription::BranchDescription(
			std::string const& moduleLabel, 
			std::string const& processName, 
			std::string const& name, 
			std::string const& fName, 
			std::string const& pin, 
			ModuleDescriptionID const& mdID,
			std::set<ParameterSetID> const& psetIDs,
			std::set<std::string> const& aliases) :
    moduleLabel_(moduleLabel),
    processName_(processName),
    productID_(),
    fullClassName_(name),
    friendlyClassName_(fName),
    productInstanceName_(pin),
    moduleDescriptionID_(mdID),
    psetIDs_(psetIDs),
    branchAliases_(aliases),
    produced_(true) {
    init();
  }

  void
  BranchDescription::init() const {
    char const underscore('_');
    char const period('.');

    if (friendlyClassName_.find(underscore) != std::string::npos) {
      throw cms::Exception("IllegalCharacter") << "Class name '" << friendlyClassName_
      << "' contains an underscore ('_'), which is illegal in the name of a product.\n";
    }

    if (moduleLabel_.find(underscore) != std::string::npos) {
      throw cms::Exception("IllegalCharacter") << "Module label '" << moduleLabel_
      << "' contains an underscore ('_'), which is illegal in a module label.\n";
    }

    if (productInstanceName_.find(underscore) != std::string::npos) {
      throw cms::Exception("IllegalCharacter") << "Product instance name '" << productInstanceName_
      << "' contains an underscore ('_'), which is illegal in a product instance name.\n";
    }

    if (processName_.find(underscore) != std::string::npos) {
      throw cms::Exception("IllegalCharacter") << "Process name '" << processName_
      << "' contains an underscore ('_'), which is illegal in a process name.\n";
    }

    branchName_ = friendlyClassName_ + underscore + moduleLabel_ + underscore +
      productInstanceName_ + underscore + processName_ + period;
  }

  void
  BranchDescription::write(std::ostream& os) const {
    os << "Process Name = " << processName_ << std::endl;
    os << "ModuleLabel = " << moduleLabel_ << std::endl;
    os << "Product ID = " << productID_ << '\n';
    os << "Class Name = " << fullClassName_ << '\n';
    os << "Friendly Class Name = " << friendlyClassName_ << '\n';
    os << "Product Instance Name = " << productInstanceName_ << std::endl;
  }

  bool
  operator<(BranchDescription const& a, BranchDescription const& b) {
    if (a.friendlyClassName_ < b.friendlyClassName_) return true;
    if (b.friendlyClassName_ < a.friendlyClassName_) return false;
    if (a.productInstanceName_ < b.productInstanceName_) return true;
    if (b.productInstanceName_ < a.productInstanceName_) return false;
    if (a.processName_ < b.processName_) return true;
    if (b.processName_ < a.processName_) return false;
    if (a.moduleLabel_ < b.moduleLabel_) return true;
    if (b.moduleLabel_ < a.moduleLabel_) return false;
    if (a.fullClassName_ < b.fullClassName_) return true;
    if (b.fullClassName_ < a.fullClassName_) return false;
    if (a.productID_ < b.productID_) return true;
    return false;
  }

  bool
  operator==(BranchDescription const& a, BranchDescription const& b) {
    return !(a < b || b < a);
  }
}
