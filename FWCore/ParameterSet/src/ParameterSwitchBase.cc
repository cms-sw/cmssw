
#include "FWCore/ParameterSet/interface/ParameterSwitchBase.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <utility>

namespace edm {

  ParameterSwitchBase::~ParameterSwitchBase() { }

  void
  ParameterSwitchBase::
  throwDuplicateCaseValues(std::string const& switchLabel) const {
    throw edm::Exception(errors::LogicError)
      << "When adding a ParameterSwitch to a ParameterSetDescription the values\n"
      << "associated with the different cases must be unique.  Duplicate\n"
      << "values were found for the switch with label: \"" << switchLabel 
      << "\"\n";
  }

  void
  ParameterSwitchBase::
  insertAndCheckLabels(std::string const& switchLabel,
                       std::set<std::string> & usedLabels,
                       std::set<std::string> & labels) const {
        
    std::pair<std::set<std::string>::iterator,bool> status = labels.insert(switchLabel);
    if (status.second == false) {
      throw edm::Exception(errors::LogicError)
        << "The label used for the switch parameter in a ParameterSetDescription\n"
        << "must be different from the labels used in the associated cases.  The following\n"
        << "duplicate label was found: \"" << switchLabel << "\"\n";
    }
    usedLabels.insert(labels.begin(), labels.end());
  }


  void
  ParameterSwitchBase::
  insertAndCheckTypes(ParameterTypes switchType,
                      std::set<ParameterTypes> const& caseParameterTypes,
                      std::set<ParameterTypes> const& caseWildcardTypes,
                      std::set<ParameterTypes> & parameterTypes,
                      std::set<ParameterTypes> & wildcardTypes) const {
    
    if (caseWildcardTypes.find(switchType) != caseWildcardTypes.end()) {
      throw edm::Exception(errors::LogicError)
        << "The type used for the switch parameter in a ParameterSetDescription\n"
        << "must be different from the types used for wildcards in the associated cases.  The following\n"
        << "duplicate type was found: \"" << parameterTypeEnumToString(switchType) << "\"\n";
    }
    parameterTypes.insert(switchType);
    parameterTypes.insert(caseParameterTypes.begin(), caseParameterTypes.end());
    wildcardTypes.insert(caseWildcardTypes.begin(), caseWildcardTypes.end());
  }

  void
  ParameterSwitchBase::
  throwNoCaseForDefault(std::string const& switchLabel) const {
    throw edm::Exception(errors::LogicError)
      << "The default value used for the switch parameter in a ParameterSetDescription\n"
      << "must match the value used to select one of the associated cases.  This is not\n"
      << "true for the switch named \"" << switchLabel << "\"\n";
  }

  void
  ParameterSwitchBase::
  throwNoCaseForSwitchValue(std::string const& message) const {
    throw edm::Exception(errors::Configuration)
      << message;
  }

  bool
  ParameterSwitchBase::
  partiallyExists_(ParameterSet const& pset) const {
    return exists(pset);
  }

  int
  ParameterSwitchBase::
  howManyExclusiveOrSubNodesExist_(ParameterSet const& pset) const {
    return exists(pset) ? 1 : 0; 
  }
}
