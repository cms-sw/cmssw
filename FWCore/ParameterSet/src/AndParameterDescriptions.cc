
#include "FWCore/ParameterSet/interface/AndParameterDescriptions.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "boost/bind.hpp"

#include <algorithm>
#include <sstream>

namespace edm {

  AndParameterDescriptions::
  AndParameterDescriptions(ParameterDescriptionNode const& node_left,
                           ParameterDescriptionNode const& node_right) :
    node_left_(node_left.clone()),
    node_right_(node_right.clone()) {
  }

  AndParameterDescriptions::
  AndParameterDescriptions(std::auto_ptr<ParameterDescriptionNode> node_left,
                           ParameterDescriptionNode const& node_right) :
    node_left_(node_left),
    node_right_(node_right.clone()) {
  }

  AndParameterDescriptions::
  AndParameterDescriptions(ParameterDescriptionNode const& node_left,
                           std::auto_ptr<ParameterDescriptionNode> node_right) :
    node_left_(node_left.clone()),
    node_right_(node_right) {
  }

  AndParameterDescriptions::
  AndParameterDescriptions(std::auto_ptr<ParameterDescriptionNode> node_left,
                           std::auto_ptr<ParameterDescriptionNode> node_right) :
    node_left_(node_left),
    node_right_(node_right) {
  }

  void
  AndParameterDescriptions::
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
  AndParameterDescriptions::
  validate_(ParameterSet & pset,
            std::set<std::string> & validatedLabels,
            bool optional) const {
    if (partiallyExists(pset) || !optional) {
      node_left_->validate(pset, validatedLabels, false);
      node_right_->validate(pset, validatedLabels, false);
    }
  }

  void
  AndParameterDescriptions::
  writeCfi_(std::ostream & os,
            bool & startWithComma,
            int indentation,
            bool & wroteSomething) const {
    node_left_->writeCfi(os, startWithComma, indentation, wroteSomething);
    node_right_->writeCfi(os, startWithComma, indentation, wroteSomething);
  }

  bool
  AndParameterDescriptions::
  exists_(ParameterSet const& pset) const {
    return node_left_->exists(pset) && node_right_->exists(pset);
  }

  bool
  AndParameterDescriptions::
  partiallyExists_(ParameterSet const& pset) const {
    return node_left_->exists(pset) || node_right_->exists(pset);
  }

  int
  AndParameterDescriptions::
  howManyExclusiveOrSubNodesExist_(ParameterSet const& pset) const {
    return exists(pset) ? 1 : 0; 
  }

  void
  AndParameterDescriptions::
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
  AndParameterDescriptions::
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
          << "Types used for wildcards in different nodes of a ParameterSetDescription\n"
          << "\"and\" expression must be different from types used for other parameters.\n"
          << "The following duplicate types were detected:\n"
          << ss.str()
          << "\n";
      }
    }
  }
}
