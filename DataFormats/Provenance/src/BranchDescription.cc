#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/FriendlyName.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"
#include <ostream>
#include <sstream>
#include <stdlib.h>

/*----------------------------------------------------------------------


----------------------------------------------------------------------*/

namespace edm {
  BranchDescription::Transients::Transients() :
    parameterSetID_(),
    branchName_(),
    wrappedName_(),
    produced_(false),
    present_(true),
    transient_(false),
    type_(),
    splitLevel_(),
    basketSize_(),
    parameterSetIDs_(),
    moduleNames_() {
   }

  BranchDescription::BranchDescription() :
    branchType_(InEvent),
    moduleLabel_(),
    processName_(),
    branchID_(),
    productID_(),
    fullClassName_(),
    friendlyClassName_(),
    productInstanceName_(),
    branchAliases_(),
    transients_()
  {
    // do not call init here! It will result in an exception throw.
  }

  BranchDescription::BranchDescription(
			BranchType const& branchType,
			std::string const& mdLabel, 
			std::string const& procName, 
			std::string const& name, 
			std::string const& fName, 
			std::string const& pin,
			ModuleDescription const& modDesc,
			std::set<std::string> const& aliases) :
    branchType_(branchType),
    moduleLabel_(mdLabel),
    processName_(procName),
    branchID_(),
    productID_(),
    fullClassName_(name),
    friendlyClassName_(fName),
    productInstanceName_(pin),
    branchAliases_(aliases),
    transients_()
  {
    present() = true;
    produced() = true;
    transients_.get().parameterSetID_ = modDesc.parameterSetID();
    parameterSetIDs().insert(std::make_pair(modDesc.processConfigurationID(),modDesc.parameterSetID()));
    moduleNames().insert(std::make_pair(modDesc.processConfigurationID(),modDesc.moduleName()));
    init();
  }

  void
  BranchDescription::init() const {
    if (!branchName().empty()) {
      return;	// already called
    }
    throwIfInvalid_();

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

    branchName().reserve(friendlyClassName().size() +
			moduleLabel().size() +
			productInstanceName().size() +
			processName().size() + 4);
    branchName() += friendlyClassName();
    branchName() += underscore;
    branchName() += moduleLabel();
    branchName() += underscore;
    branchName() += productInstanceName();
    branchName() += underscore;
    branchName() += processName();
    branchName() += period;

    if (!branchID_.isValid()) {
      branchID_.setID(branchName());
    }

    Reflex::Type t = Reflex::Type::ByName(fullClassName());
    Reflex::PropertyList p = t.Properties();
    transient() = (p.HasProperty("persistent") ? p.PropertyAsString("persistent") == std::string("false") : false);

    wrappedName() = wrappedClassName(fullClassName());
    type() = Reflex::Type::ByName(wrappedName());
    Reflex::PropertyList wp = type().Properties();
    if (wp.HasProperty("splitLevel")) {
	splitLevel() = strtol(wp.PropertyAsString("splitLevel").c_str(), 0, 0);
	if (splitLevel() < 0) {
          throw cms::Exception("IllegalSplitLevel") << "' An illegal ROOT split level of " <<
	  splitLevel() << " is specified for class " << wrappedName() << ".'\n";
	}
	++splitLevel(); //Compensate for wrapper
    } else {
	splitLevel() = invalidSplitLevel; 
    }
    if (wp.HasProperty("basketSize")) {
	basketSize() = strtol(wp.PropertyAsString("basketSize").c_str(), 0, 0);
	if (basketSize() <= 0) {
          throw cms::Exception("IllegalBasketSize") << "' An illegal ROOT basket size of " <<
	  basketSize() << " is specified for class " << wrappedName() << "'.\n";
	}
    } else {
	basketSize() = invalidBasketSize; 
    }
  }

  ParameterSetID const&
    BranchDescription::psetID() const {
    assert(!parameterSetIDs().empty());
    if (parameterSetIDs().size() != 1) {
      throw cms::Exception("Ambiguous")
	<< "Your application requires all events on Branch '" << branchName()
	<< "'\n to have the same provenance. This file has events with mixed provenance\n"
	<< "on this branch.  Use a different input file.\n";
    }
    return parameterSetIDs().begin()->second;
  }

  void
  BranchDescription::merge(BranchDescription const& other) {
    parameterSetIDs().insert(other.parameterSetIDs().begin(), other.parameterSetIDs().end());
    moduleNames().insert(other.moduleNames().begin(), other.moduleNames().end());
    branchAliases_.insert(other.branchAliases().begin(), other.branchAliases().end());
    present() = present() || other.present();
    if (splitLevel() == invalidSplitLevel) splitLevel() = other.splitLevel();
    if (basketSize() == invalidBasketSize) basketSize() = other.basketSize();
  }

  void
  BranchDescription::write(std::ostream& os) const {
    os << "Branch Type = " << branchType() << std::endl;
    os << "Process Name = " << processName() << std::endl;
    os << "ModuleLabel = " << moduleLabel() << std::endl;
    os << "Branch ID = " << branchID() << '\n';
    os << "Class Name = " << fullClassName() << '\n';
    os << "Friendly Class Name = " << friendlyClassName() << '\n';
    os << "Product Instance Name = " << productInstanceName() << std::endl;
  }

  void throwExceptionWithText(const char* txt)
  {
    edm::Exception e(edm::errors::LogicError);
    e << "Problem using an incomplete BranchDescription\n"
      << txt 
      << "\nPlease report this error to the FWCore developers";
    throw e;
  }

  void
  BranchDescription::throwIfInvalid_() const
  {
    if (branchType_ >= edm::NumBranchTypes)
      throwExceptionWithText("Illegal BranchType detected");

    if (moduleLabel_.empty())
      throwExceptionWithText("Module label is not allowed to be empty");

    if (processName_.empty())
      throwExceptionWithText("Process name is not allowed to be empty");

    if (fullClassName_.empty())
      throwExceptionWithText("Full class name is not allowed to be empty");

    if (friendlyClassName_.empty())
      throwExceptionWithText("Friendly class name is not allowed to be empty");

    if (produced() && !parameterSetID().isValid())
      throwExceptionWithText("Invalid ParameterSetID detected");    
  }

  void
  BranchDescription::updateFriendlyClassName() {
    friendlyClassName_ = friendlyname::friendlyName(fullClassName());
  }

  bool
  operator<(BranchDescription const& a, BranchDescription const& b) {
    if (a.processName() < b.processName()) return true;
    if (b.processName() < a.processName()) return false;
    if (a.fullClassName() < b.fullClassName()) return true;
    if (b.fullClassName() < a.fullClassName()) return false;
    if (a.friendlyClassName() < b.friendlyClassName()) return true;
    if (b.friendlyClassName() < a.friendlyClassName()) return false;
    if (a.productInstanceName() < b.productInstanceName()) return true;
    if (b.productInstanceName() < a.productInstanceName()) return false;
    if (a.moduleLabel() < b.moduleLabel()) return true;
    if (b.moduleLabel() < a.moduleLabel()) return false;
    if (a.branchType() < b.branchType()) return true;
    if (b.branchType() < a.branchType()) return false;
    if (a.branchID() < b.branchID()) return true;
    if (b.branchID() < a.branchID()) return false;
    if (a.parameterSetIDs() < b.parameterSetIDs()) return true;
    if (b.parameterSetIDs() < a.parameterSetIDs()) return false;
    if (a.moduleNames() < b.moduleNames()) return true;
    if (b.moduleNames() < a.moduleNames()) return false;
    if (a.branchAliases() < b.branchAliases()) return true;
    if (b.branchAliases() < a.branchAliases()) return false;
    if (a.present() < b.present()) return true;
    if (b.present() < a.present()) return false;
    return false;
  }

  bool
  combinable(BranchDescription const& a, BranchDescription const& b) {
    return
    (a.branchType() == b.branchType()) &&
    (a.processName() == b.processName()) &&
    (a.fullClassName() == b.fullClassName()) &&
    (a.friendlyClassName() == b.friendlyClassName()) &&
    (a.productInstanceName() == b.productInstanceName()) &&
    (a.moduleLabel() == b.moduleLabel()) &&
    (a.branchID() == b.branchID());
  }

  bool
  operator==(BranchDescription const& a, BranchDescription const& b) {
    return combinable(a, b) &&
       (a.present() == b.present()) &&
       (a.moduleNames() == b.moduleNames()) &&
       (a.parameterSetIDs() == b.parameterSetIDs()) &&
       (a.branchAliases() == b.branchAliases());
  }

  std::string
  match(BranchDescription const& a, BranchDescription const& b,
	std::string const& fileName,
	BranchDescription::MatchMode m) {
    std::ostringstream differences;
    if (a.branchName() != b.branchName()) {
      differences << "Branch name '" << b.branchName() << "' does not match '" << a.branchName() << "'.\n";
      // Need not compare components of branch name individually.
      // (a.friendlyClassName() != b.friendlyClassName())
      // (a.moduleLabel() != b.moduleLabel())
      // (a.productInstanceName() != b.productInstanceName())
      // (a.processName() != b.processName())
    }
    if (a.branchType() != b.branchType()) {
      differences << "Branch '" << b.branchName() << "' is a(n) '" << b.branchType() << "' branch\n";
      differences << "    in file '" << fileName << "', but a(n) '" << a.branchType() << "' branch in previous files.\n";
    }
    if (a.branchID() != b.branchID()) {
      differences << "Branch '" << b.branchName() << "' has a branch ID of '" << b.branchID() << "'\n";
      differences << "    in file '" << fileName << "', but '" << a.branchID() << "' in previous files.\n";
    }
    if (a.fullClassName() != b.fullClassName()) {
      differences << "Products on branch '" << b.branchName() << "' have type '" << b.fullClassName() << "'\n";
      differences << "    in file '" << fileName << "', but '" << a.fullClassName() << "' in previous files.\n";
    }
    if (b.present() && !a.present()) {
      differences << "Branch '" << a.branchName() << "' was dropped in previous files but is present in '" << fileName << "'.\n";
    }
    if (m == BranchDescription::Strict) {
	if (b.parameterSetIDs().size() > 1) {
	  differences << "Branch '" << b.branchName() << "' uses more than one parameter set in file '" << fileName << "'.\n";
	} else if (a.parameterSetIDs().size() > 1) {
	  differences << "Branch '" << a.branchName() << "' uses more than one parameter set in previous files.\n";
	} else if (a.parameterSetIDs() != b.parameterSetIDs()) {
	  differences << "Branch '" << b.branchName() << "' uses different parameter sets in file '" << fileName << "'.\n";
	  differences << "    than in previous files.\n";
	}
    }
    return differences.str();
  }
}
