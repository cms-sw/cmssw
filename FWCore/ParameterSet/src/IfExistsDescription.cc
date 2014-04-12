
#include "FWCore/ParameterSet/interface/IfExistsDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/DocFormatHelper.h"

#include <algorithm>
#include <sstream>
#include <ostream>
#include <iomanip>

namespace edm {

  IfExistsDescription::
  IfExistsDescription(ParameterDescriptionNode const& node_left,
                      ParameterDescriptionNode const& node_right) :
    node_left_(node_left.clone()),
    node_right_(node_right.clone()) {
  }

  IfExistsDescription::
  IfExistsDescription(std::auto_ptr<ParameterDescriptionNode> node_left,
                      ParameterDescriptionNode const& node_right) :
    node_left_(node_left),
    node_right_(node_right.clone()) {
  }

  IfExistsDescription::
  IfExistsDescription(ParameterDescriptionNode const& node_left,
                      std::auto_ptr<ParameterDescriptionNode> node_right) :
    node_left_(node_left.clone()),
    node_right_(node_right) {
  }

  IfExistsDescription::
  IfExistsDescription(std::auto_ptr<ParameterDescriptionNode> node_left,
                      std::auto_ptr<ParameterDescriptionNode> node_right) :
    node_left_(node_left),
    node_right_(node_right) {
  }

  void
  IfExistsDescription::
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
  IfExistsDescription::
  validate_(ParameterSet & pset,
            std::set<std::string> & validatedLabels,
            bool optional) const {

    bool leftExists = node_left_->exists(pset);
    bool rightExists = node_right_->exists(pset);

    if (!leftExists && !rightExists) {
      return;
    }
    else if (leftExists && rightExists) {
      node_left_->validate(pset, validatedLabels, false);
      node_right_->validate(pset, validatedLabels, false);
    }
    else if (leftExists && !rightExists) {
      node_left_->validate(pset, validatedLabels, false);
      if (!optional) node_right_->validate(pset, validatedLabels, false);
    }
    else if (!leftExists && rightExists) {
      node_left_->validate(pset, validatedLabels, false);
      node_right_->validate(pset, validatedLabels, false);
    }
  }

  void
  IfExistsDescription::
  writeCfi_(std::ostream & os,
            bool & startWithComma,
            int indentation,
            bool & wroteSomething) const {
    node_left_->writeCfi(os, startWithComma, indentation, wroteSomething);
    node_right_->writeCfi(os, startWithComma, indentation, wroteSomething);
  }

  void
  IfExistsDescription::
  print_(std::ostream & os,
         bool optional,
         bool writeToCfi,
         DocFormatHelper & dfh) {

    if (dfh.pass() == 1) {

      dfh.indent(os);
      os << "IfExists pair:";

      if (dfh.brief()) {

        if (optional)  os << " optional";

        if (!writeToCfi) os << " (do not write to cfi)";

        os << " see Section " << dfh.section() << "." << dfh.counter() << "\n";
      }
      // not brief
      else {

        os << "\n";
        dfh.indent2(os);

        if (optional)  os << "optional";
        if (!writeToCfi) os << " (do not write to cfi)";
        if (optional || !writeToCfi) {
          os << "\n";
          dfh.indent2(os);
        }

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
  IfExistsDescription::
  printNestedContent_(std::ostream & os,
                      bool optional,
                      DocFormatHelper & dfh) {

    int indentation = dfh.indentation();
    if (dfh.parent() != DocFormatHelper::TOP) {
      indentation -= DocFormatHelper::offsetSectionContent();
    }

    std::stringstream ss;
    ss << dfh.section() << "." << dfh.counter();
    std::string newSection = ss.str();

    printSpaces(os, indentation);
    os << "Section " << newSection;
    if (optional) os << " optional";
    os << " IfExists pair description:\n";
    printSpaces(os, indentation);
    if (optional) {
      os << "If the first parameter exists, then the second is allowed to exist\n";
    }
    else {
      os << "If the first parameter exists, then the second is required to exist\n";
    }
    if (!dfh.brief()) os << "\n";

    DocFormatHelper new_dfh(dfh);
    new_dfh.init();
    new_dfh.setSection(newSection);
    new_dfh.setIndentation(indentation + DocFormatHelper::offsetSectionContent());
    new_dfh.setParent(DocFormatHelper::OTHER);

    node_left_->print(os, false, true, new_dfh);
    node_right_->print(os, false, true, new_dfh);

    new_dfh.setPass(1);
    new_dfh.setCounter(0);

    node_left_->print(os, false, true, new_dfh);
    node_right_->print(os, false, true, new_dfh);

    new_dfh.setPass(2);
    new_dfh.setCounter(0);

    node_left_->printNestedContent(os, false, new_dfh);
    node_right_->printNestedContent(os, false , new_dfh);
  }

  bool
  IfExistsDescription::
  exists_(ParameterSet const& pset) const {
    bool leftExists = node_left_->exists(pset);
    bool rightExists = node_right_->exists(pset);

    if (leftExists && rightExists) return true;
    else if (!leftExists && !rightExists) return true;
    return false;
  }

  bool
  IfExistsDescription::
  partiallyExists_(ParameterSet const& pset) const {
    return exists(pset);
  }

  int
  IfExistsDescription::
  howManyXORSubNodesExist_(ParameterSet const& pset) const {
    return exists(pset) ? 1 : 0;
  }

  void
  IfExistsDescription::
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
        << "Labels used in a node of a ParameterSetDescription\n"
        << "\"ifExists\" expression must be not be the same as labels used\n"
        << "in other nodes of the expression.  The following duplicate\n"
        << "labels were detected:\n"
        << ss.str()
        << "\n";
    }
  }

  void
  IfExistsDescription::
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
          << "Types used for wildcards in a node of a ParameterSetDescription\n"
          << "\"ifExists\" expression must be different from types used for other parameters\n"
          << "in other nodes.  The following duplicate types were detected:\n"
          << ss.str()
          << "\n";
      }
    }
  }
}
