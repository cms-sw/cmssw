
#include "FWCore/ParameterSet/interface/ParameterSwitchBase.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/DocFormatHelper.h"

#include <ostream>
#include <iomanip>
#include <sstream>

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

  void
  ParameterSwitchBase::
  printBase(std::ostream & os,
            bool optional,
            bool writeToCfi,
            DocFormatHelper & dfh,
            std::string const& switchLabel,
            bool isTracked,
            std::string const& typeString) const {

    if (dfh.pass() == 0) {
      dfh.setAtLeast1(switchLabel.size() + 9U);
      if (isTracked) {
        dfh.setAtLeast2(typeString.size());
      }
      else {
        dfh.setAtLeast2(typeString.size() + 10U);
      }
      dfh.setAtLeast3(8U);
    }
    if (dfh.pass() == 1) {

      dfh.indent(os);

      if (dfh.brief()) {

	std::stringstream ss;
        ss << switchLabel << " (switch)"; 
        os << std::left << std::setw(dfh.column1()) << ss.str();
        os << " ";

        os << std::setw(dfh.column2());
        if (isTracked) {
          os << typeString;
        }
        else {
          std::stringstream ss1;
          ss1 << "untracked " << typeString;
          os << ss1.str();
        }

        os << " " << std::setw(dfh.column3());
        if (optional)  os << "optional";
        else  os << "";

        if (!writeToCfi) os << " (do not write to cfi)";

        os << " see Section " << dfh.section() << "." << dfh.counter() << "\n";
      }
      // not brief
      else {

        os << switchLabel << " (switch)\n";

        dfh.indent2(os);
        os << "type: ";
        if (!isTracked) os << "untracked ";
        os << typeString << " ";

        if (optional)  os << "optional";

        if (!writeToCfi) os << " (do not write to cfi)";
        os << "\n";

        dfh.indent2(os);
        os << "see Section " << dfh.section() << "." << dfh.counter() << "\n";

        if (!comment().empty()) {
          DocFormatHelper::wrapAndPrintText(os,
                                            comment(),
                                            dfh.startColumn2(),
                                            dfh.commentWidth());
        }
        os << "\n";
      }
    }
  }

  bool
  ParameterSwitchBase::
  hasNestedContent_() {
    return true;
  }

  void
  ParameterSwitchBase::
  printNestedContentBase(std::ostream & os,
                         DocFormatHelper & dfh,
                         DocFormatHelper & new_dfh,
                         std::string const& switchLabel) {

    int indentation = dfh.indentation();
    if (dfh.parent() != DocFormatHelper::TOP) {
      indentation -= DocFormatHelper::offsetSectionContent();
    }

    std::stringstream ss;
    ss << dfh.section() << "." << dfh.counter();
    std::string newSection = ss.str();

    os << std::setfill(' ') << std::setw(indentation) << "";
    os << "Section " << newSection
       << " " << switchLabel << " (switch):\n";
    
    if (!dfh.brief()) {
      os << std::setfill(' ') << std::setw(indentation) << "";
      os << "The value of \"" << switchLabel << "\" controls which other parameters\n";
      os << std::setfill(' ') << std::setw(indentation) << "";
      os << "are required or allowed to be in the PSet.\n";
    }
    if (!dfh.brief()) os << "\n";

    new_dfh.init();
    new_dfh.setSection(newSection);
    new_dfh.setIndentation(indentation + DocFormatHelper::offsetSectionContent());
    new_dfh.setParent(DocFormatHelper::OTHER);
  }


  void
  ParameterSwitchBase::
  printCase(std::pair<bool, edm::value_ptr<ParameterDescriptionNode> > const& p,
            std::ostream & os,
            bool optional,
            DocFormatHelper & dfh,
            std::string const& switchLabel) {
    if (dfh.pass() == 0) {
      p.second->print(os, false, true, dfh);
    }
    if (dfh.pass() == 1) {
      dfh.indent(os);
      os << "if " << switchLabel << " = ";
      if (p.first) os << "True";
      else os << "False";
      os << "\n";
      p.second->print(os, false, true, dfh);
    }
    if (dfh.pass() == 2) {
      p.second->printNestedContent(os, false, dfh);
    }
  }

  void
  ParameterSwitchBase::
  printCase(std::pair<int, edm::value_ptr<ParameterDescriptionNode> > const& p,
            std::ostream & os,
            bool optional,
            DocFormatHelper & dfh,
            std::string const& switchLabel) {
    if (dfh.pass() == 0) {
      p.second->print(os, false, true, dfh);
    }
    if (dfh.pass() == 1) {
      dfh.indent(os);
      os << "if " << switchLabel << " = " << p.first << "\n";
      p.second->print(os, false, true, dfh);
    }
    if (dfh.pass() == 2) {
      p.second->printNestedContent(os, false, dfh);
    }
  }

  void
  ParameterSwitchBase::
  printCase(std::pair<std::string, edm::value_ptr<ParameterDescriptionNode> > const& p,
            std::ostream & os,
            bool optional,
            DocFormatHelper & dfh,
            std::string const& switchLabel) {
    if (dfh.pass() == 0) {
      p.second->print(os, false, true, dfh);
    }
    if (dfh.pass() == 1) {
      dfh.indent(os);
      os << "if " << switchLabel << " = \"" << p.first << "\"\n";
      p.second->print(os, false, true, dfh);
    }
    if (dfh.pass() == 2) {
      p.second->printNestedContent(os, false, dfh);
    }
  }

  bool
  ParameterSwitchBase::
  partiallyExists_(ParameterSet const& pset) const {
    return exists(pset);
  }

  int
  ParameterSwitchBase::
  howManyXORSubNodesExist_(ParameterSet const& pset) const {
    return exists(pset) ? 1 : 0; 
  }
}
