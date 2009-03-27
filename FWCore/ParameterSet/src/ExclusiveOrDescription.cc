
#include "FWCore/ParameterSet/interface/ExclusiveOrDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {

  ExclusiveOrDescription::
  ExclusiveOrDescription(ParameterDescriptionNode const& node_left,
                         ParameterDescriptionNode const& node_right) :
    node_left_(node_left.clone()),
    node_right_(node_right.clone()) {
  }

  ExclusiveOrDescription::
  ExclusiveOrDescription(std::auto_ptr<ParameterDescriptionNode> node_left,
                         ParameterDescriptionNode const& node_right) :
    node_left_(node_left),
    node_right_(node_right.clone()) {
  }

  ExclusiveOrDescription::
  ExclusiveOrDescription(ParameterDescriptionNode const& node_left,
                         std::auto_ptr<ParameterDescriptionNode> node_right) :
    node_left_(node_left.clone()),
    node_right_(node_right) {
  }

  ExclusiveOrDescription::
  ExclusiveOrDescription(std::auto_ptr<ParameterDescriptionNode> node_left,
                         std::auto_ptr<ParameterDescriptionNode> node_right) :
    node_left_(node_left),
    node_right_(node_right) {
  }

  void
  ExclusiveOrDescription::
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
  ExclusiveOrDescription::
  validate_(ParameterSet & pset,
            std::set<std::string> & validatedLabels,
            bool optional) const {

    int nExistLeft = node_left_->howManyExclusiveOrSubNodesExist(pset);
    int nExistRight = node_right_->howManyExclusiveOrSubNodesExist(pset);
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
      nExistLeft = node_left_->howManyExclusiveOrSubNodesExist(pset);
      nExistRight = node_right_->howManyExclusiveOrSubNodesExist(pset);
      nTotal = nExistLeft + nExistRight;

      if (nTotal != 1) {
        throwAfterValidation();
      }
    }
  }

  void
  ExclusiveOrDescription::
  writeCfi_(std::ostream & os,
            bool & startWithComma,
            int indentation,
            bool & wroteSomething) const {
    node_left_->writeCfi(os, startWithComma, indentation, wroteSomething);
  }

  bool
  ExclusiveOrDescription::
  exists_(ParameterSet const& pset) const {
    int nTotal = node_left_->howManyExclusiveOrSubNodesExist(pset) +
                 node_right_->howManyExclusiveOrSubNodesExist(pset);
    return nTotal == 1;
  }

  bool
  ExclusiveOrDescription::
  partiallyExists_(ParameterSet const& pset) const {
    return exists(pset);
  }

  int
  ExclusiveOrDescription::
  howManyExclusiveOrSubNodesExist_(ParameterSet const& pset) const {
    return node_left_->howManyExclusiveOrSubNodesExist(pset) +
           node_right_->howManyExclusiveOrSubNodesExist(pset);
  }

  void
  ExclusiveOrDescription::
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
  ExclusiveOrDescription::
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
