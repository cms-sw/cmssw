#include "DataFormats/Common/interface/BranchDescription.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Utilities/interface/Exception.h"

/*----------------------------------------------------------------------

$Id: BranchDescription.cc,v 1.11 2006/08/01 05:34:43 wmtan Exp $

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
    processConfigurationIDs_(),
    branchAliases_(),
    produced_(false),
    present_(true)
  { }

  BranchDescription::BranchDescription(
			std::string const& moduleLabel, 
			std::string const& processName, 
			std::string const& name, 
			std::string const& fName, 
			std::string const& pin, 
			ModuleDescriptionID const& mdID,
			std::set<ParameterSetID> const& psetIDs,
			std::set<ProcessConfigurationID> const& procConfigIDs,
			std::set<std::string> const& aliases) :
    moduleLabel_(moduleLabel),
    processName_(processName),
    productID_(),
    fullClassName_(name),
    friendlyClassName_(fName),
    productInstanceName_(pin),
    moduleDescriptionID_(mdID),
    psetIDs_(psetIDs),
    processConfigurationIDs_(procConfigIDs),
    branchAliases_(aliases),
    produced_(true),
    present_(true) {
    init();
  }

  void
  BranchDescription::init() const {
    char const underscore('_');
    char const period('.');

    if (friendlyClassName_.find(underscore) != std::string::npos) {
      throw cms::Exception("IllegalCharacter") << "Class name '" << friendlyClassName()
      << "' contains an underscore ('_'), which is illegal in the name of a product.\n";
    }

    if (moduleLabel_.find(underscore) != std::string::npos) {
      throw cms::Exception("IllegalCharacter") << "Module label '" << moduleLabel()
      << "' contains an underscore ('_'), which is illegal in a module label.\n";
    }

    if (productInstanceName_.find(underscore) != std::string::npos) {
      throw cms::Exception("IllegalCharacter") << "Product instance name '" << productInstanceName()
      << "' contains an underscore ('_'), which is illegal in a product instance name.\n";
    }

    if (processName_.find(underscore) != std::string::npos) {
      throw cms::Exception("IllegalCharacter") << "Process name '" << processName()
      << "' contains an underscore ('_'), which is illegal in a process name.\n";
    }

    branchName_ = friendlyClassName() + underscore + moduleLabel() + underscore +
      productInstanceName() + underscore + processName() + period;
  }

  ParameterSetID const&
    BranchDescription::psetID() const {
    assert(!psetIDs().empty());
    if (psetIDs().size() != 1) {
      throw cms::Exception("Ambiguous")
	<< "Your application requires all events on Branch '" << branchName()
	<< "'\n to have the same provenance. This file has events with mixed provenance\n"
	<< "on this branch.  Use a different input file.\n";
    }
    return *psetIDs().begin();
  }

  bool
  BranchDescription::merge(BranchDescription const& other, MatchMode m) {
    if (!match(*this, other, m)) return false;
    psetIDs_.insert(other.psetIDs().begin(), other.psetIDs().end());
    processConfigurationIDs_.insert(other.processConfigurationIDs().begin(), other.processConfigurationIDs().end());
    branchAliases_.insert(other.branchAliases().begin(), other.branchAliases().end());
    return true;
  }

  void
  BranchDescription::write(std::ostream& os) const {
    os << "Process Name = " << processName() << std::endl;
    os << "ModuleLabel = " << moduleLabel() << std::endl;
    os << "Product ID = " << productID() << '\n';
    os << "Class Name = " << fullClassName() << '\n';
    os << "Friendly Class Name = " << friendlyClassName() << '\n';
    os << "Product Instance Name = " << productInstanceName() << std::endl;
  }

  bool
  operator<(BranchDescription const& a, BranchDescription const& b) {
    if (a.processName() < b.processName()) return true;
    if (b.processName() < a.processName()) return false;
    if (a.productID() < b.productID()) return true;
    if (b.productID() < a.productID()) return false;
    if (a.fullClassName() < b.fullClassName()) return true;
    if (b.fullClassName() < a.fullClassName()) return false;
    if (a.friendlyClassName() < b.friendlyClassName()) return true;
    if (b.friendlyClassName() < a.friendlyClassName()) return false;
    if (a.productInstanceName() < b.productInstanceName()) return true;
    if (b.productInstanceName() < a.productInstanceName()) return false;
    if (a.moduleLabel() < b.moduleLabel()) return true;
    if (b.moduleLabel() < a.moduleLabel()) return false;
    if (a.psetIDs() < b.psetIDs()) return true;
    if (b.psetIDs() < a.psetIDs()) return false;
    if (a.processConfigurationIDs() < b.processConfigurationIDs()) return true;
    if (b.processConfigurationIDs() < a.processConfigurationIDs()) return false;
    if (a.branchAliases() < b.branchAliases()) return true;
    if (b.branchAliases() < a.branchAliases()) return false;
    if (a.present() < b.present()) return true;
    if (b.present() < a.present()) return false;
    return false;
  }

  bool
  operator==(BranchDescription const& a, BranchDescription const& b) {
    return
    (a.processName() == b.processName()) &&
    (a.productID() == b.productID()) &&
    (a.fullClassName() == b.fullClassName()) &&
    (a.friendlyClassName() == b.friendlyClassName()) &&
    (a.productInstanceName() == b.productInstanceName()) &&
    (a.moduleLabel() == b.moduleLabel()) &&
    (a.psetIDs() == b.psetIDs()) &&
    (a.processConfigurationIDs() == b.processConfigurationIDs()) &&
    (a.branchAliases() == b.branchAliases()) &&
    (a.present() == b.present());
  }

  bool
  match(BranchDescription const& a, BranchDescription const& b, BranchDescription::MatchMode m) {
    bool looseMatch = 
      (a.processName() == b.processName()) &&
      (a.productID() == b.productID()) &&
      (a.fullClassName() == b.fullClassName()) &&
      (a.friendlyClassName() == b.friendlyClassName()) &&
      (a.productInstanceName() == b.productInstanceName()) &&
      (a.moduleLabel() == b.moduleLabel()) &&
      (a.present() == b.present());
    if (m == BranchDescription::Strict) {
      return (looseMatch &&
		 a.psetIDs().size() == 1 &&
		 a.processConfigurationIDs().size() == 1 &&
		 a.psetIDs() == b.psetIDs() &&
		 a.processConfigurationIDs() == b.processConfigurationIDs());
    }
    return looseMatch;
  }
}
