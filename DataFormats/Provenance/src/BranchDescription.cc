#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/FriendlyName.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"

#include <ostream>
#include <sstream>
#include <stdlib.h>

class TClass;
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
    parameterSetIDs_(),
    moduleNames_(),
    transient_(false),
    wrappedType_(),
    unwrappedType_(),
    wrapperInterfaceBase_(0),
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
    dropped() = false;
    transient_.produced_ = produced,
    onDemand() = false;
    transient_.moduleName_ = moduleName;
    transient_.parameterSetID_ = parameterSetID;
    unwrappedType() = theTypeWithDict;
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
    dropped() = false;
    transient_.produced_ = aliasForBranch.produced(),
    onDemand() = aliasForBranch.onDemand();
    transient_.moduleName_ = aliasForBranch.moduleName();
    transient_.parameterSetID_ = aliasForBranch.parameterSetID();
    unwrappedType() = aliasForBranch.unwrappedType();
    init();
  }

  void
  BranchDescription::initBranchName() const {
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

    if(!branchID_.isValid()) {
      branchID_.setID(branchName());
    }
  }

  void
  BranchDescription::initFromDictionary() const {
    if(bool(wrappedType())) {
      return;  // already initialized;
    }

    throwIfInvalid_();

    wrappedName() = wrappedClassName(fullClassName());

    // unwrapped type.
    unwrappedType() = TypeWithDict::byName(fullClassName());
    if(!bool(unwrappedType())) {
      splitLevel() = invalidSplitLevel;
      basketSize() = invalidBasketSize;
      transient() = false;
      return;
    }

    wrappedType() = TypeWithDict::byName(wrappedName());
    if(!bool(wrappedType())) {
      splitLevel() = invalidSplitLevel;
      basketSize() = invalidBasketSize;
      return;
    }
    Reflex::PropertyList wp = Reflex::Type::ByTypeInfo(wrappedType().typeInfo()).Properties();
    transient() = (wp.HasProperty("persistent") ? wp.PropertyAsString("persistent") == std::string("false") : false);
    if(transient()) {
      splitLevel() = invalidSplitLevel;
      basketSize() = invalidBasketSize;
      return;
    }
    if(wp.HasProperty("splitLevel")) {
      splitLevel() = strtol(wp.PropertyAsString("splitLevel").c_str(), 0, 0);
      if(splitLevel() < 0) {
        throw cms::Exception("IllegalSplitLevel") << "' An illegal ROOT split level of " <<
        splitLevel() << " is specified for class " << wrappedName() << ".'\n";
      }
      ++splitLevel(); //Compensate for wrapper
    } else {
      splitLevel() = invalidSplitLevel;
    }
    if(wp.HasProperty("basketSize")) {
      basketSize() = strtol(wp.PropertyAsString("basketSize").c_str(), 0, 0);
      if(basketSize() <= 0) {
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
    if(parameterSetIDs().size() != 1) {
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
    if(splitLevel() == invalidSplitLevel) splitLevel() = other.splitLevel();
    if(basketSize() == invalidBasketSize) basketSize() = other.basketSize();
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
    branchName().clear();
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
    if(a.parameterSetIDs() < b.parameterSetIDs()) return true;
    if(b.parameterSetIDs() < a.parameterSetIDs()) return false;
    if(a.moduleNames() < b.moduleNames()) return true;
    if(b.moduleNames() < a.moduleNames()) return false;
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
       (a.moduleNames() == b.moduleNames()) &&
       (a.parameterSetIDs() == b.parameterSetIDs()) &&
       (a.branchAliases() == b.branchAliases());
  }

  std::string
  match(BranchDescription const& a, BranchDescription const& b,
        std::string const& fileName,
        BranchDescription::MatchMode m) {
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
    if(m == BranchDescription::Strict) {
        if(b.parameterSetIDs().size() > 1) {
          differences << "Branch '" << b.branchName() << "' uses more than one parameter set in file '" << fileName << "'.\n";
        } else if(a.parameterSetIDs().size() > 1) {
          differences << "Branch '" << a.branchName() << "' uses more than one parameter set in previous files.\n";
        } else if(a.parameterSetIDs() != b.parameterSetIDs()) {
          differences << "Branch '" << b.branchName() << "' uses different parameter sets in file '" << fileName << "'.\n";
          differences << "    than in previous files.\n";
        }
    }
    return differences.str();
  }

  WrapperInterfaceBase const*
  BranchDescription::getInterface() const {
    if(wrapperInterfaceBase() == 0) {
      // This could be done in init(), but we only want to do it on demand, for performance reasons.
      TypeWithDict type = TypeWithDict::byName(wrappedName());
      type.invokeByName(wrapperInterfaceBase(), "getInterface");
      assert(wrapperInterfaceBase() != 0);
    }
    return wrapperInterfaceBase();
  }
}
