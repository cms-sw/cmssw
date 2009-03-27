
#include "FWCore/ParameterSet/interface/ParameterWildcardBase.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {

  ParameterWildcardBase::~ParameterWildcardBase() { }

  ParameterWildcardBase::ParameterWildcardBase(ParameterTypes iType,
                                               bool isTracked,
                                               WildcardValidationCriteria criteria)
    :type_(iType),
     isTracked_(isTracked),
     criteria_(criteria)
  { }

  void
  ParameterWildcardBase::
  throwIfInvalidPattern(char const* pattern) const {
    std::string sPattern(pattern);
    throwIfInvalidPattern(sPattern);
  }

  void
  ParameterWildcardBase::
  throwIfInvalidPattern(std::string const& pattern) const {
    if (pattern != std::string("*")) {
      throw edm::Exception(errors::Configuration)
        << "Currently, the only supported wildcard in ParameterSetDescriptions\n"
        << "is the single character \"*\".  The configuration contains a wildcard\n"
        << "with pattern \"" << pattern << "\" and type \"" << parameterTypeEnumToString(type()) << "\"\n"
        << "At some future date, globbing or regular expression support may be added\n"
        << "if there are any requests for it from users.\n";
    }
  }

  void
  ParameterWildcardBase::
  validateMatchingNames(std::vector<std::string> const& matchingNames,
                        std::set<std::string> & validatedLabels,
                        bool optional) const {
    validatedLabels.insert(matchingNames.begin(), matchingNames.end());
    if (optional || criteria_ == RequireZeroOrMore) return;
    if (criteria_ == RequireAtLeastOne && matchingNames.size() < 1U) {
      throw edm::Exception(errors::Configuration)
        << "Parameter wildcard of type \"" << parameterTypeEnumToString(type()) << "\" requires\n"
        << "at least one match and there are no parameters in the configuration matching\n"
        << "that type.\n";
    }
    else if (criteria_ == RequireExactlyOne && matchingNames.size() != 1U) {
      throw edm::Exception(errors::Configuration)
        << "Parameter wildcard of type \"" << parameterTypeEnumToString(type()) << "\" requires\n"
        << "exactly one match and there are " << matchingNames.size() << " matching parameters\n"
        << "in the configuration.\n";
    }
  }

  void
  ParameterWildcardBase::
  checkAndGetLabelsAndTypes_(std::set<std::string> & usedLabels,
                             std::set<ParameterTypes> & parameterTypes,
                             std::set<ParameterTypes> & wildcardTypes) const {
    wildcardTypes.insert(type());
  }

  void
  ParameterWildcardBase::
  writeCfi_(std::ostream & os,
            bool & startWithComma,
            int indentation,
            bool & wroteSomething) const {
    // Until we implement default labels and values there is nothing
    // to do here.
  }

  bool
  ParameterWildcardBase::
  partiallyExists_(ParameterSet const& pset) const {
    return exists(pset);
  }

  int
  ParameterWildcardBase::
  howManyExclusiveOrSubNodesExist_(ParameterSet const& pset) const {
    return exists(pset) ? 1 : 0; 
  }
}
