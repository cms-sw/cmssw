#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/FriendlyName.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"

#include <cassert>
#include <ostream>
#include <sstream>
#include <cstdlib>

/*----------------------------------------------------------------------


----------------------------------------------------------------------*/

namespace edm {
  BranchDescription::Transients::Transients() :
    parameterSetID_(),
    moduleName_(),
    branchName_(),
    wrappedName_(),
    produced_(false),
    onDemand_(false),
    dropped_(false),
    transient_(false),
    wrappedType_(),
    unwrappedType_(),
    splitLevel_(),
    basketSize_() {
   }

  void
  BranchDescription::Transients::reset() {
    *this = BranchDescription::Transients();
  }

  BranchDescription::BranchDescription() :
    branchType_(InEvent),
    moduleLabel_(),
    processName_(),
    branchID_(),
    fullClassName_(),
    friendlyClassName_(),
    productInstanceName_(),
    branchAliases_(),
    aliasForBranchID_(),
    transient_() {
    // do not call init here! It will result in an exception throw.
  }

  BranchDescription::BranchDescription(
                        BranchType const& branchType,
                        std::string const& moduleLabel,
                        std::string const& processName,
                        std::string const& className,
                        std::string const& friendlyClassName,
                        std::string const& productInstanceName,
                        std::string const& moduleName,
                        ParameterSetID const& parameterSetID,
                        TypeWithDict const& theTypeWithDict,
                        bool produced,
                        std::set<std::string> const& aliases) :
      branchType_(branchType),
      moduleLabel_(moduleLabel),
      processName_(processName),
      branchID_(),
      fullClassName_(className),
      friendlyClassName_(friendlyClassName),
      productInstanceName_(productInstanceName),
      branchAliases_(aliases),
      transient_() {
    setDropped(false);
    setProduced(produced);
    setOnDemand(false);
    transient_.moduleName_ = moduleName;
    transient_.parameterSetID_ = parameterSetID;
    setUnwrappedType(theTypeWithDict);
    init();
  }

  BranchDescription::BranchDescription(
                        BranchDescription const& aliasForBranch,
                        std::string const& moduleLabelAlias,
                        std::string const& productInstanceAlias) :
      branchType_(aliasForBranch.branchType()),
      moduleLabel_(moduleLabelAlias),
      processName_(aliasForBranch.processName()),
      branchID_(),
      fullClassName_(aliasForBranch.className()),
      friendlyClassName_(aliasForBranch.friendlyClassName()),
      productInstanceName_(productInstanceAlias),
      branchAliases_(aliasForBranch.branchAliases()),
      aliasForBranchID_(aliasForBranch.branchID()),
      transient_() {
    setDropped(false);
    setProduced(aliasForBranch.produced());
    setOnDemand(aliasForBranch.onDemand());
    transient_.moduleName_ = aliasForBranch.moduleName();
    transient_.parameterSetID_ = aliasForBranch.parameterSetID();
    setUnwrappedType(aliasForBranch.unwrappedType());
    init();
  }

  void
  BranchDescription::initBranchName() {
    if(!branchName().empty()) {
      return;  // already called
    }
    throwIfInvalid_();

    char const underscore('_');
    char const period('.');

    if(friendlyClassName_.find(underscore) != std::string::npos) {
      throw cms::Exception("IllegalCharacter") << "Class name '" << friendlyClassName()
      << "' contains an underscore ('_'), which is illegal in the name of a product.\n";
    }

    if(moduleLabel_.find(underscore) != std::string::npos) {
      throw cms::Exception("IllegalCharacter") << "Module label '" << moduleLabel()
      << "' contains an underscore ('_'), which is illegal in a module label.\n";
    }

    if(productInstanceName_.find(underscore) != std::string::npos) {
      throw cms::Exception("IllegalCharacter") << "Product instance name '" << productInstanceName()
      << "' contains an underscore ('_'), which is illegal in a product instance name.\n";
    }

    if(processName_.find(underscore) != std::string::npos) {
      throw cms::Exception("IllegalCharacter") << "Process name '" << processName()
      << "' contains an underscore ('_'), which is illegal in a process name.\n";
    }

    std::string& brName = transient_.branchName_;
    brName.reserve(friendlyClassName().size() +
                   moduleLabel().size() +
                   productInstanceName().size() +
                   processName().size() + 4);
    brName += friendlyClassName();
    brName += underscore;
    brName += moduleLabel();
    brName += underscore;
    brName += productInstanceName();
    brName += underscore;
    brName += processName();
    brName += period;

    if(!branchID_.isValid()) {
      branchID_.setID(brName);
    }
  }

  void
  BranchDescription::initFromDictionary() {
    if(bool(wrappedType())) {
      return;  // already initialized;
    }

    throwIfInvalid_();

    try {
      setWrappedName(wrappedClassName(fullClassName()));
      
      // unwrapped type.
      setUnwrappedType(TypeWithDict::byName(fullClassName()));
      if(!bool(unwrappedType())) {
	setSplitLevel(invalidSplitLevel);
	setBasketSize(invalidBasketSize);
	setTransient(false);
	return;
      }


      setWrappedType(TypeWithDict::byName(wrappedName()));
      if(!bool(wrappedType())) {
	setSplitLevel(invalidSplitLevel);
	setBasketSize(invalidBasketSize);
	return;
      }
    } catch( edm::Exception& caughtException) {
      caughtException.addContext(std::string{"While initializing meta data for branch: "}+branchName());
      throw;
    }
    Reflex::PropertyList wp = Reflex::Type::ByTypeInfo(wrappedType().typeInfo()).Properties();
    setTransient((wp.HasProperty("persistent") ? wp.PropertyAsString("persistent") == std::string("false") : false));
    if(transient()) {
      setSplitLevel(invalidSplitLevel);
      setBasketSize(invalidBasketSize);
      return;
    }
    if(wp.HasProperty("splitLevel")) {
      setSplitLevel(strtol(wp.PropertyAsString("splitLevel").c_str(), 0, 0));
      if(splitLevel() < 0) {
        throw cms::Exception("IllegalSplitLevel") << "' An illegal ROOT split level of " <<
        splitLevel() << " is specified for class " << wrappedName() << ".'\n";
      }
      setSplitLevel(splitLevel() + 1); //Compensate for wrapper
    } else {
      setSplitLevel(invalidSplitLevel);
    }
    if(wp.HasProperty("basketSize")) {
      setBasketSize(strtol(wp.PropertyAsString("basketSize").c_str(), 0, 0));
      if(basketSize() <= 0) {
        throw cms::Exception("IllegalBasketSize") << "' An illegal ROOT basket size of " <<
        basketSize() << " is specified for class " << wrappedName() << "'.\n";
      }
    } else {
      setBasketSize(invalidBasketSize);
    }
  }

  void
  BranchDescription::merge(BranchDescription const& other) {
    branchAliases_.insert(other.branchAliases().begin(), other.branchAliases().end());
    if(splitLevel() == invalidSplitLevel) setSplitLevel(other.splitLevel());
    if(basketSize() == invalidBasketSize) setBasketSize(other.basketSize());
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

  void throwExceptionWithText(char const* txt) {
    Exception e(errors::LogicError);
    e << "Problem using an incomplete BranchDescription\n"
      << txt
      << "\nPlease report this error to the FWCore developers";
    throw e;
  }

  void
  BranchDescription::throwIfInvalid_() const {
    if(branchType_ >= NumBranchTypes)
      throwExceptionWithText("Illegal BranchType detected");

    if(moduleLabel_.empty())
      throwExceptionWithText("Module label is not allowed to be empty");

    if(processName_.empty())
      throwExceptionWithText("Process name is not allowed to be empty");

    if(fullClassName_.empty())
      throwExceptionWithText("Full class name is not allowed to be empty");

    if(friendlyClassName_.empty())
      throwExceptionWithText("Friendly class name is not allowed to be empty");

    if(produced() && !parameterSetID().isValid())
      throwExceptionWithText("Invalid ParameterSetID detected");
  }

  void
  BranchDescription::updateFriendlyClassName() {
    friendlyClassName_ = friendlyname::friendlyName(fullClassName());
    clearBranchName();
    initBranchName();
  }

  bool
  operator<(BranchDescription const& a, BranchDescription const& b) {
    if(a.processName() < b.processName()) return true;
    if(b.processName() < a.processName()) return false;
    if(a.fullClassName() < b.fullClassName()) return true;
    if(b.fullClassName() < a.fullClassName()) return false;
    if(a.friendlyClassName() < b.friendlyClassName()) return true;
    if(b.friendlyClassName() < a.friendlyClassName()) return false;
    if(a.productInstanceName() < b.productInstanceName()) return true;
    if(b.productInstanceName() < a.productInstanceName()) return false;
    if(a.moduleLabel() < b.moduleLabel()) return true;
    if(b.moduleLabel() < a.moduleLabel()) return false;
    if(a.branchType() < b.branchType()) return true;
    if(b.branchType() < a.branchType()) return false;
    if(a.branchID() < b.branchID()) return true;
    if(b.branchID() < a.branchID()) return false;
    if(a.branchAliases() < b.branchAliases()) return true;
    if(b.branchAliases() < a.branchAliases()) return false;
    if(a.present() < b.present()) return true;
    if(b.present() < a.present()) return false;
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
       (a.dropped() == b.dropped()) &&
       (a.branchAliases() == b.branchAliases());
  }

  std::string
  match(BranchDescription const& a, BranchDescription const& b,
        std::string const& fileName) {
    std::ostringstream differences;
    if(a.branchName() != b.branchName()) {
      differences << "Branch name '" << b.branchName() << "' does not match '" << a.branchName() << "'.\n";
      // Need not compare components of branch name individually.
      // (a.friendlyClassName() != b.friendlyClassName())
      // (a.moduleLabel() != b.moduleLabel())
      // (a.productInstanceName() != b.productInstanceName())
      // (a.processName() != b.processName())
    }
    if(a.branchType() != b.branchType()) {
      differences << "Branch '" << b.branchName() << "' is a(n) '" << b.branchType() << "' branch\n";
      differences << "    in file '" << fileName << "', but a(n) '" << a.branchType() << "' branch in previous files.\n";
    }
    if(a.branchID() != b.branchID()) {
      differences << "Branch '" << b.branchName() << "' has a branch ID of '" << b.branchID() << "'\n";
      differences << "    in file '" << fileName << "', but '" << a.branchID() << "' in previous files.\n";
    }
    if(a.fullClassName() != b.fullClassName()) {
      differences << "Products on branch '" << b.branchName() << "' have type '" << b.fullClassName() << "'\n";
      differences << "    in file '" << fileName << "', but '" << a.fullClassName() << "' in previous files.\n";
    }
    if(!b.dropped() && a.dropped()) {
      differences << "Branch '" << a.branchName() << "' was dropped in the first input file but is present in '" << fileName << "'.\n";
    }
    return differences.str();
  }
}
