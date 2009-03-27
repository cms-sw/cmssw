
#include "FWCore/ParameterSet/interface/OrParameterDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <algorithm>
#include <sstream>

namespace edm {

  OrParameterDescription::
  OrParameterDescription(ParameterDescriptionNode const& node_left,
                         ParameterDescriptionNode const& node_right) :
    node_left_(node_left.clone()),
    node_right_(node_right.clone()) {
  }

  OrParameterDescription::
  OrParameterDescription(std::auto_ptr<ParameterDescriptionNode> node_left,
                         ParameterDescriptionNode const& node_right) :
    node_left_(node_left),
    node_right_(node_right.clone()) {
  }

  OrParameterDescription::
  OrParameterDescription(ParameterDescriptionNode const& node_left,
                         std::auto_ptr<ParameterDescriptionNode> node_right) :
    node_left_(node_left.clone()),
    node_right_(node_right) {
  }

  OrParameterDescription::
  OrParameterDescription(std::auto_ptr<ParameterDescriptionNode> node_left,
                         std::auto_ptr<ParameterDescriptionNode> node_right) :
    node_left_(node_left),
    node_right_(node_right) {
  }

  void
  OrParameterDescription::
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
  OrParameterDescription::
  validate_(ParameterSet & pset,
            std::set<std::string> & validatedLabels,
            bool optional) const {

    bool leftExists = node_left_->exists(pset);
    bool rightExists = node_right_->exists(pset);

    if (leftExists || rightExists) {
      if (leftExists) node_left_->validate(pset, validatedLabels, false);
      if (rightExists) node_right_->validate(pset, validatedLabels, false);
      return;
    }

    node_left_->validate(pset, validatedLabels, false);
    if (node_right_->exists(pset)) node_right_->validate(pset, validatedLabels, false);
  }

  void
  OrParameterDescription::
  writeCfi_(std::ostream & os,
            bool & startWithComma,
            int indentation,
            bool & wroteSomething) const {
    node_left_->writeCfi(os, startWithComma, indentation, wroteSomething);
  }

  bool
  OrParameterDescription::
  exists_(ParameterSet const& pset) const {
    return node_left_->exists(pset) ||
           node_right_->exists(pset);
  }

  bool
  OrParameterDescription::
  partiallyExists_(ParameterSet const& pset) const {
    return exists(pset);
  }

  int
  OrParameterDescription::
  howManyExclusiveOrSubNodesExist_(ParameterSet const& pset) const {
    return exists(pset) ? 1 : 0;
  }

  void
  OrParameterDescription::
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
        << "\"or\" expression must be not be the same as labels used\n"
        << "in other nodes of the expression.  The following duplicate\n"
        << "labels were detected:\n"
        << ss.str()
        << "\n";
    }
  }

  void
  OrParameterDescription::
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
          ss << " \"" << *iter <<  "\"\n";
        }
        throw edm::Exception(errors::LogicError)
          << "Types used for wildcards in a node of a ParameterSetDescription\n"
          << "\"or\" expression must be different from types used for other parameters\n"
          << "in other nodes.  The following duplicate types were detected:\n"
          << ss.str()
          << "\n";
      }
    }
  }
}
