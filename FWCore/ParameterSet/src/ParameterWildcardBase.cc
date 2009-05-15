
#include "FWCore/ParameterSet/interface/ParameterWildcardBase.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/DocFormatHelper.h"

#include <ostream>
#include <iomanip>
#include <sstream>

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
    if (criteria_ == RequireZeroOrMore) return;
    if (criteria_ == RequireAtLeastOne && matchingNames.size() < 1U && !optional) {
      throw edm::Exception(errors::Configuration)
        << "Parameter wildcard of type \"" << parameterTypeEnumToString(type()) << "\" requires "
        << "at least one match\n"
        << "and there are no parameters in the configuration matching\n"
        << "that type.\n";
    }
    else if (criteria_ == RequireExactlyOne) {
      if ( (matchingNames.size() < 1U && !optional) ||
           matchingNames.size() > 1U) {
        throw edm::Exception(errors::Configuration)
          << "Parameter wildcard of type \"" << parameterTypeEnumToString(type()) << "\" requires\n"
          << "exactly one match and there are " << matchingNames.size() << " matching parameters\n"
          << "in the configuration.\n";
      }
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
  print_(std::ostream & os,
         bool optional,
	 bool writeToCfi,
         DocFormatHelper & dfh)
  {
    if (dfh.pass() == 0) {
      dfh.setAtLeast1(11U);
      if (isTracked()) {
        dfh.setAtLeast2(parameterTypeEnumToString(type()).size());
      }
      else {
        dfh.setAtLeast2(parameterTypeEnumToString(type()).size() + 10U);
      }
      dfh.setAtLeast3(8U);
    }
    else {
      
      if (dfh.brief()) {

        dfh.indent(os);
        os << std::left << std::setw(dfh.column1()) << "wildcard: *" << " ";

        if (isTracked()) {
          os << std::setw(dfh.column2()) << parameterTypeEnumToString(type());
        }
        else {
	  std::stringstream ss;
          ss << "untracked " << parameterTypeEnumToString(type());
          os << ss.str();
        }

        os << " ";
        os << std::setw(dfh.column3());
        if (optional)  os << "optional";
        else  os << "";

        if (criteria() == RequireZeroOrMore) {
          os << " (require zero or more)";
        }
        else if (criteria() == RequireAtLeastOne) {
          os << " (require at least one)";
        }
        else if (criteria() == RequireExactlyOne) {
          os << " (require exactly one)";
        }
        os << "\n";
        if (hasNestedContent()) {
          dfh.indent(os);
          os << "  (see Section " << dfh.section()
             << "." << dfh.counter() << ")\n";
        }
      }
      // not brief
      else {

        dfh.indent(os);
        os << "labels must match this wildcard pattern: *\n";

        dfh.indent2(os);
        os << "type: ";
        if (isTracked()) {
          os << parameterTypeEnumToString(type());
        }
        else {
          os << "untracked " << parameterTypeEnumToString(type());
        }

        if (optional)  os << " optional";
        os << "\n";

        dfh.indent2(os);
        os << "criteria: ";
        if (criteria() == RequireZeroOrMore) os << "require zero or more";
        else if (criteria() == RequireAtLeastOne) os << "require at least one";
        else if (criteria() == RequireExactlyOne) os << "require exactly one";
        os << "\n";

        if (hasNestedContent()) {
          dfh.indent2(os);
          os << "(see Section " << dfh.section()
             << "." << dfh.counter() << ")\n";
        }

        if (!comment().empty()) {
          DocFormatHelper::wrapAndPrintText(os,
                                            comment(),
                                            dfh.startColumn2(),
                                            dfh.commentWidth());
        }
        os << "\n";
      }
      os << std::right;
    }
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
  howManyXORSubNodesExist_(ParameterSet const& pset) const {
    return exists(pset) ? 1 : 0; 
  }
}
