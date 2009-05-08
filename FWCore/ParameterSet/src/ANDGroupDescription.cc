
#include "FWCore/ParameterSet/interface/ANDGroupDescription.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/DocFormatHelper.h"

#include "boost/bind.hpp"

#include <algorithm>
#include <sstream>
#include <ostream>
#include <iomanip>

namespace edm {

  ANDGroupDescription::
  ANDGroupDescription(ParameterDescriptionNode const& node_left,
                      ParameterDescriptionNode const& node_right) :
    node_left_(node_left.clone()),
    node_right_(node_right.clone()) {
  }

  ANDGroupDescription::
  ANDGroupDescription(std::auto_ptr<ParameterDescriptionNode> node_left,
                      ParameterDescriptionNode const& node_right) :
    node_left_(node_left),
    node_right_(node_right.clone()) {
  }

  ANDGroupDescription::
  ANDGroupDescription(ParameterDescriptionNode const& node_left,
                      std::auto_ptr<ParameterDescriptionNode> node_right) :
    node_left_(node_left.clone()),
    node_right_(node_right) {
  }

  ANDGroupDescription::
  ANDGroupDescription(std::auto_ptr<ParameterDescriptionNode> node_left,
                      std::auto_ptr<ParameterDescriptionNode> node_right) :
    node_left_(node_left),
    node_right_(node_right) {
  }

  void
  ANDGroupDescription::
  checkAndGetLabelsAndTypes_(std::set<std::string> & usedLabels,
                             std::set<ParameterTypes> & parameterTypes,
                             std::set<ParameterTypes> & wildcardTypes) const {

    std::set<std::string> labelsLeft;
    std::set<ParameterTypes> parameterTypesLeft;
    std::set<ParameterTypes> wildcardTypesLeft;
    node_left_->checkAndGetLabelsAndTypes(labelsLeft, parameterTypesLeft, wildcardTypesLeft);

    std::set<std::string> labelsRight;
    std::set<ParameterTypes> parameterTypesRight;
    std::set<ParameterTypes> wildcardTypesRight;
    node_right_->checkAndGetLabelsAndTypes(labelsRight, parameterTypesRight, wildcardTypesRight);

    throwIfDuplicateLabels(labelsLeft, labelsRight);
    throwIfDuplicateTypes(wildcardTypesLeft, parameterTypesRight);
    throwIfDuplicateTypes(wildcardTypesRight, parameterTypesLeft);

    usedLabels.insert(labelsLeft.begin(), labelsLeft.end());
    usedLabels.insert(labelsRight.begin(), labelsRight.end());

    parameterTypes.insert(parameterTypesRight.begin(), parameterTypesRight.end());
    parameterTypes.insert(parameterTypesLeft.begin(), parameterTypesLeft.end());

    wildcardTypes.insert(wildcardTypesRight.begin(), wildcardTypesRight.end());
    wildcardTypes.insert(wildcardTypesLeft.begin(), wildcardTypesLeft.end());
  }

  void
  ANDGroupDescription::
  validate_(ParameterSet & pset,
            std::set<std::string> & validatedLabels,
            bool optional) const {
    if (partiallyExists(pset) || !optional) {
      node_left_->validate(pset, validatedLabels, false);
      node_right_->validate(pset, validatedLabels, false);
    }
  }

  void
  ANDGroupDescription::
  writeCfi_(std::ostream & os,
            bool & startWithComma,
            int indentation,
            bool & wroteSomething) const {
    node_left_->writeCfi(os, startWithComma, indentation, wroteSomething);
    node_right_->writeCfi(os, startWithComma, indentation, wroteSomething);
  }

  void
  ANDGroupDescription::
  print_(std::ostream & os,
         bool optional,
         bool writeToCfi,
         DocFormatHelper & dfh) {

    if (dfh.parent() == DocFormatHelper::AND) {
      dfh.decrementCounter();
      node_left_->print(os, false, true, dfh);
      node_right_->print(os, false, true, dfh);
      return;
    }

    if (dfh.pass() == 1) {

      dfh.indent(os);
      os << "AND group:";

      if (dfh.brief()) {

        if (optional)  os << " optional";
        else  os << " required";

        if (!writeToCfi) os << " (do not write to cfi)";

        os << " see Section " << dfh.section() << "." << dfh.counter() << "\n";
      }
      // not brief
      else {

        os << "\n";
        dfh.indent2(os);
        if (optional)  os << "optional";
        else  os << "required";

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

  void
  ANDGroupDescription::
  printNestedContent_(std::ostream & os,
                      bool optional,
                      DocFormatHelper & dfh) {

    if (dfh.parent() == DocFormatHelper::AND) {
      dfh.decrementCounter();
      node_left_->printNestedContent(os, false, dfh);
      node_right_->printNestedContent(os, false, dfh);
      return;
    }

    int indentation = dfh.indentation();
    if (dfh.parent() != DocFormatHelper::TOP) {
      indentation -= DocFormatHelper::offsetSectionContent();
    }

    std::stringstream ss;
    ss << dfh.section() << "." << dfh.counter();
    std::string newSection = ss.str();

    os << std::setfill(' ') << std::setw(indentation) << "";
    os << "Section " << newSection
       << " AND group description:\n";
    os << std::setfill(' ') << std::setw(indentation) << "";
    if (optional) {
      os << "This optional AND group requires all or none of the following to be in the PSet\n";
    }
    else {
      os << "This AND group requires all of the following to be in the PSet\n";
    }
    if (!dfh.brief()) os << "\n";

    DocFormatHelper new_dfh(dfh);
    new_dfh.init();
    new_dfh.setSection(newSection);
    new_dfh.setIndentation(indentation + DocFormatHelper::offsetSectionContent());
    new_dfh.setParent(DocFormatHelper::AND);

    node_left_->print(os, false, true, new_dfh);
    node_right_->print(os, false, true, new_dfh);

    new_dfh.setPass(1);
    new_dfh.setCounter(0);

    node_left_->print(os, false, true, new_dfh);
    node_right_->print(os, false, true, new_dfh);

    new_dfh.setPass(2);
    new_dfh.setCounter(0);

    node_left_->printNestedContent(os, false, new_dfh);
    node_right_->printNestedContent(os, false, new_dfh);
  }

  bool
  ANDGroupDescription::
  exists_(ParameterSet const& pset) const {
    return node_left_->exists(pset) && node_right_->exists(pset);
  }

  bool
  ANDGroupDescription::
  partiallyExists_(ParameterSet const& pset) const {
    return node_left_->partiallyExists(pset) || node_right_->partiallyExists(pset);
  }

  int
  ANDGroupDescription::
  howManyXORSubNodesExist_(ParameterSet const& pset) const {
    return exists(pset) ? 1 : 0; 
  }

  void
  ANDGroupDescription::
  throwIfDuplicateLabels(std::set<std::string> const& labelsLeft,
                         std::set<std::string> const& labelsRight) const {

    std::set<std::string> duplicateLabels;
    std::insert_iterator<std::set<std::string> > insertIter(duplicateLabels, duplicateLabels.begin());
    std::set_intersection(labelsLeft.begin(), labelsLeft.end(),
                          labelsRight.begin(), labelsRight.end(),
                          insertIter);
    if (!duplicateLabels.empty()) {
      std::stringstream ss;
      for (std::set<std::string>::const_iterator iter = duplicateLabels.begin(),
	                                         iEnd = duplicateLabels.end();
           iter != iEnd;
           ++iter) {
        ss << " \"" << *iter <<  "\"\n";
      }
      throw edm::Exception(errors::LogicError)
        << "Labels used in different nodes of a ParameterSetDescription\n"
        << "\"and\" expression must be unique.  The following duplicate\n"
        << "labels were detected:\n"
        << ss.str()
        << "\n";
    }
  }

  void
  ANDGroupDescription::
  throwIfDuplicateTypes(std::set<ParameterTypes> const& types1,
                        std::set<ParameterTypes> const& types2) const
  {
    if (!types1.empty()) {
      std::set<ParameterTypes> duplicateTypes;
      std::insert_iterator<std::set<ParameterTypes> > insertIter(duplicateTypes, duplicateTypes.begin());
      std::set_intersection(types1.begin(), types1.end(),
                            types2.begin(), types2.end(),
                            insertIter);
      if (!duplicateTypes.empty()) {
        std::stringstream ss;
        for (std::set<ParameterTypes>::const_iterator iter = duplicateTypes.begin(),
	                                              iEnd = duplicateTypes.end();
             iter != iEnd;
             ++iter) {
          ss << " \"" << parameterTypeEnumToString(*iter) <<  "\"\n";
        }
        throw edm::Exception(errors::LogicError)
          << "Types used for wildcards in different nodes of a ParameterSetDescription\n"
          << "\"and\" expression must be different from types used for other parameters.\n"
          << "The following duplicate types were detected:\n"
          << ss.str()
          << "\n";
      }
    }
  }
}
