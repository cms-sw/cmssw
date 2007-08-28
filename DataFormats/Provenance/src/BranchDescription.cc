#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"
#include <ostream>
#include <sstream>
#include <stdlib.h>

/*----------------------------------------------------------------------

$Id: BranchDescription.cc,v 1.3 2007/08/23 23:32:53 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  BranchDescription::BranchDescription() :
    branchType_(InEvent),
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
    branchName_(),
    produced_(false),
    present_(true),
    provenancePresent_(true),
    transient_(false),
    type_(),
    splitLevel_(invalidSplitLevel),
    basketSize_(invalidBasketSize)
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
    productID_(),
    fullClassName_(name),
    friendlyClassName_(fName),
    productInstanceName_(pin),
    moduleDescriptionID_(modDesc.id()),
    psetIDs_(),
    processConfigurationIDs_(),
    branchAliases_(aliases),
    branchName_(),
    produced_(true),
    present_(true),
    provenancePresent_(true),
    transient_(false),
    type_(),
    splitLevel_(invalidSplitLevel),
    basketSize_(invalidBasketSize)
  {
    psetIDs_.insert(modDesc.parameterSetID());
    processConfigurationIDs_.insert(modDesc.processConfigurationID());
    init();
  }



  BranchDescription::BranchDescription(
			BranchType const& branchType,
			std::string const& mdLabel, 
			std::string const& procName, 
			std::string const& name, 
			std::string const& fName, 
			std::string const& pin, 
			ModuleDescriptionID const& mdID,
			std::set<ParameterSetID> const& psIDs,
			std::set<ProcessConfigurationID> const& procConfigIDs,
			std::set<std::string> const& aliases) :
    branchType_(branchType),
    moduleLabel_(mdLabel),
    processName_(procName),
    productID_(),
    fullClassName_(name),
    friendlyClassName_(fName),
    productInstanceName_(pin),
    moduleDescriptionID_(mdID),
    psetIDs_(psIDs),
    processConfigurationIDs_(procConfigIDs),
    branchAliases_(aliases),
    branchName_(),
    produced_(true),
    present_(true),
    provenancePresent_(true),
    transient_(false),
    type_(),
    splitLevel_(invalidSplitLevel),
    basketSize_(invalidBasketSize)
  {
    init();
  }

  void
  BranchDescription::init() const {
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

    branchName_ = friendlyClassName() + underscore + moduleLabel() + underscore +
      productInstanceName() + underscore + processName() + period;

    ROOT::Reflex::Type t = ROOT::Reflex::Type::ByName(fullClassName());
    ROOT::Reflex::PropertyList p = t.Properties();
    transient_ = (p.HasProperty("persistent") ? p.PropertyAsString("persistent") == std::string("false") : false);

    type_ = ROOT::Reflex::Type::ByName(wrappedClassName(fullClassName()));
    ROOT::Reflex::PropertyList wp = type_.Properties();
    if (wp.HasProperty("splitLevel")) {
	splitLevel_ = strtol(wp.PropertyAsString("splitLevel").c_str(), 0, 0);
	if (splitLevel_ < 0) {
          throw cms::Exception("IllegalSplitLevel") << "' An illegal ROOT split level of " <<
	  splitLevel_ << " is specified for class " << wrappedClassName(fullClassName()) << ".'\n";
	}
	++splitLevel_; //Compensate for wrapper
    }
    if (wp.HasProperty("basketSize")) {
	basketSize_ = strtol(wp.PropertyAsString("basketSize").c_str(), 0, 0);
	if (basketSize_ <= 0) {
          throw cms::Exception("IllegalBasketSize") << "' An illegal ROOT basket size of " <<
	  basketSize_ << " is specified for class " << wrappedClassName(fullClassName()) << "'.\n";
	}
    }
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

  void
  BranchDescription::merge(BranchDescription const& other) {
    psetIDs_.insert(other.psetIDs().begin(), other.psetIDs().end());
    processConfigurationIDs_.insert(other.processConfigurationIDs().begin(), other.processConfigurationIDs().end());
    branchAliases_.insert(other.branchAliases().begin(), other.branchAliases().end());
  }

  void
  BranchDescription::write(std::ostream& os) const {
    os << "Branch Type = " << branchType() << std::endl;
    os << "Process Name = " << processName() << std::endl;
    os << "ModuleLabel = " << moduleLabel() << std::endl;
    os << "Product ID = " << productID() << '\n';
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
    if (branchType_ >= edm::EndBranchType)
      throwExceptionWithText("Illegal BranchType detected");

    if (moduleLabel_.empty())
      throwExceptionWithText("Module label is not allowed to be empty");

    if (processName_.empty())
      throwExceptionWithText("Process name is not allowed to be empty");

    if (fullClassName_.empty())
      throwExceptionWithText("Full class name is not allowed to be empty");

    if (friendlyClassName_.empty())
      throwExceptionWithText("Friendly class name is not allowed to be empty");

    if (produced_ && !moduleDescriptionID_.isValid())
      throwExceptionWithText("Invalid ModuleDescriptionID detected");    
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
    if (a.branchType() < b.branchType()) return true;
    if (b.branchType() < a.branchType()) return false;
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
    (a.branchType() == b.branchType()) &&
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
    if (a.productID() != b.productID()) {
      differences << "Branch '" << b.branchName() << "' has a product ID of '" << b.productID() << "'\n";
      differences << "    in file '" << fileName << "', but '" << a.productID() << "' in previous files.\n";
    }
    if (a.fullClassName() != b.fullClassName()) {
      differences << "Products on branch '" << b.branchName() << "' have type '" << b.fullClassName() << "'\n";
      differences << "    in file '" << fileName << "', but '" << a.fullClassName() << "' in previous files.\n";
    }
    if (a.present() != b.present()) {
      if (a.present()) {
	differences << "Branch '" << a.branchName() << "' was dropped in file '" << fileName << "' but is present in previous files.\n";
      } else {
	differences << "Branch '" << a.branchName() << "' was dropped in previous files but is present in '" << fileName << "'.\n";
      }
    }
    if (m == BranchDescription::Strict) {
	if (b.psetIDs().size() > 1) {
	  differences << "Branch '" << b.branchName() << "' uses more than one parameter set in file '" << fileName << "'.\n";
	} else if (a.psetIDs().size() > 1) {
	  differences << "Branch '" << a.branchName() << "' uses more than one parameter set in previous files.\n";
	} else if (a.psetIDs() != b.psetIDs()) {
	  differences << "Branch '" << b.branchName() << "' uses different parameter sets in file '" << fileName << "'.\n";
	  differences << "    than in previous files.\n";
	}

	if (b.processConfigurationIDs().size() > 1) {
	  differences << "Branch '" << b.branchName() << "' uses more than one process configuration in file '" << fileName << "'.\n";
	} else if (a.processConfigurationIDs().size() > 1) {
	  differences << "Branch '" << a.branchName() << "' uses more than one process configuration in previous files.\n";
	} else if (a.processConfigurationIDs() != b.processConfigurationIDs()) {
	  differences << "Branch '" << b.branchName() << "' uses different process configurations in file '" << fileName << "'.\n";
	  differences << "    than in previous files.\n";
	}
    }
    return differences.str();
  }
}
