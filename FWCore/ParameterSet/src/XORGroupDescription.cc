
#include "FWCore/ParameterSet/interface/XORGroupDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/DocFormatHelper.h"

#include <ostream>
#include <iomanip>

namespace edm {

  XORGroupDescription::
  XORGroupDescription(ParameterDescriptionNode const& node_left,
                      ParameterDescriptionNode const& node_right) :
    node_left_(node_left.clone()),
    node_right_(node_right.clone()) {
  }

  XORGroupDescription::
  XORGroupDescription(std::auto_ptr<ParameterDescriptionNode> node_left,
                      ParameterDescriptionNode const& node_right) :
    node_left_(node_left),
    node_right_(node_right.clone()) {
  }

  XORGroupDescription::
  XORGroupDescription(ParameterDescriptionNode const& node_left,
                      std::auto_ptr<ParameterDescriptionNode> node_right) :
    node_left_(node_left.clone()),
    node_right_(node_right) {
  }

  XORGroupDescription::
  XORGroupDescription(std::auto_ptr<ParameterDescriptionNode> node_left,
                      std::auto_ptr<ParameterDescriptionNode> node_right) :
    node_left_(node_left),
    node_right_(node_right) {
  }

  void
  XORGroupDescription::
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

    usedLabels.insert(labelsLeft.begin(), labelsLeft.end());
    usedLabels.insert(labelsRight.begin(), labelsRight.end());

    parameterTypes.insert(parameterTypesRight.begin(), parameterTypesRight.end());
    parameterTypes.insert(parameterTypesLeft.begin(), parameterTypesLeft.end());

    wildcardTypes.insert(wildcardTypesRight.begin(), wildcardTypesRight.end());
    wildcardTypes.insert(wildcardTypesLeft.begin(), wildcardTypesLeft.end());
  }

  void
  XORGroupDescription::
  validate_(ParameterSet & pset,
            std::set<std::string> & validatedLabels,
            bool optional) const {

    int nExistLeft = node_left_->howManyXORSubNodesExist(pset);
    int nExistRight = node_right_->howManyXORSubNodesExist(pset);
    int nTotal = nExistLeft + nExistRight;

    if (nTotal == 0 && optional) return;

    if (nTotal > 1) {
      throwMoreThanOneParameter();
    }

    if (nExistLeft == 1) {
      node_left_->validate(pset, validatedLabels, false);      
    }
    else if (nExistRight == 1) {
      node_right_->validate(pset, validatedLabels, false);
    }
    else if (nTotal == 0) {
      node_left_->validate(pset, validatedLabels, false);      

      // When missing parameters get inserted, both nodes could
      // be affected so we have to recheck both nodes.
      nExistLeft = node_left_->howManyXORSubNodesExist(pset);
      nExistRight = node_right_->howManyXORSubNodesExist(pset);
      nTotal = nExistLeft + nExistRight;

      if (nTotal != 1) {
        throwAfterValidation();
      }
    }
  }

  void
  XORGroupDescription::
  writeCfi_(std::ostream & os,
            bool & startWithComma,
            int indentation,
            bool & wroteSomething) const {
    node_left_->writeCfi(os, startWithComma, indentation, wroteSomething);
  }

  void
  XORGroupDescription::
  print_(std::ostream & os,
         bool optional,
         bool writeToCfi,
         DocFormatHelper & dfh) {

    if (dfh.parent() == DocFormatHelper::XOR) {
      dfh.decrementCounter();
      node_left_->print(os, false, true, dfh);
      node_right_->print(os, false, true, dfh);
      return;
    }

    if (dfh.pass() == 1) {

      dfh.indent(os);
      os << "XOR group:";

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
  XORGroupDescription::
  printNestedContent_(std::ostream & os,
                      bool optional,
                      DocFormatHelper & dfh) {

    if (dfh.parent() == DocFormatHelper::XOR) {
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

    printSpaces(os, indentation);
    os << "Section " << newSection
       << " XOR group description:\n";
    printSpaces(os, indentation);
    if (optional) {
      os << "This optional XOR group requires exactly one or none of the following to be in the PSet\n";
    }
    else {
      os << "This XOR group requires exactly one of the following to be in the PSet\n";
    }
    if (!dfh.brief()) os << "\n";

    DocFormatHelper new_dfh(dfh);
    new_dfh.init();
    new_dfh.setSection(newSection);
    new_dfh.setIndentation(indentation + DocFormatHelper::offsetSectionContent());
    new_dfh.setParent(DocFormatHelper::XOR);

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
  XORGroupDescription::
  exists_(ParameterSet const& pset) const {
    int nTotal = node_left_->howManyXORSubNodesExist(pset) +
                 node_right_->howManyXORSubNodesExist(pset);
    return nTotal == 1;
  }

  bool
  XORGroupDescription::
  partiallyExists_(ParameterSet const& pset) const {
    return exists(pset);
  }

  int
  XORGroupDescription::
  howManyXORSubNodesExist_(ParameterSet const& pset) const {
    return node_left_->howManyXORSubNodesExist(pset) +
           node_right_->howManyXORSubNodesExist(pset);
  }

  void
  XORGroupDescription::
  throwMoreThanOneParameter() const {
    // Need to expand this error message to print more information
    // I guess I need to print out the entire node structure of
    // of the xor node and all the nodes it contains.
    throw edm::Exception(errors::LogicError)
      << "Exactly one parameter can exist in a ParameterSet from a list of\n"
      << "parameters described by an \"xor\" operator in a ParameterSetDescription.\n"
      << "This rule also applies in a more general sense to the other types\n"
      << "of nodes that can appear within a ParameterSetDescription.  Only one\n"
      << "can pass validation as \"existing\".\n";
  }

  void
  XORGroupDescription::
  throwAfterValidation() const {
    // Need to expand this error message to print more information
    // I guess I need to print out the entire node structure of
    // of the xor node and all the nodes it contains.
    throw edm::Exception(errors::LogicError)
      << "Exactly one parameter can exist in a ParameterSet from a list of\n"
      << "parameters described by an \"xor\" operator in a ParameterSetDescription.\n"
      << "This rule also applies in a more general sense to the other types\n"
      << "of nodes that can appear within a ParameterSetDescription.  Only one\n"
      << "can pass validation as \"existing\".  This error has occurred after an\n"
      << "attempt to insert missing parameters to fix the problem.\n";
  }
}
