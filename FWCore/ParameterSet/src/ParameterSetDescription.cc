// -*- C++ -*-
//
// Package:     ParameterSet
// Class  :     ParameterSetDescription
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Jul 31 15:30:35 EDT 2007
//

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/IfExistsDescription.h"
#include "FWCore/ParameterSet/interface/IllegalParameters.h"
#include "FWCore/ParameterSet/interface/DocFormatHelper.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <sstream>
#include <ostream>
#include <iomanip>
#include <algorithm>

namespace edm {

  ParameterSetDescription::ParameterSetDescription() : anythingAllowed_(false), unknown_(false) {}

  ParameterSetDescription::~ParameterSetDescription() {}

  void ParameterSetDescription::setComment(std::string const& value) { comment_ = value; }

  void ParameterSetDescription::setComment(char const* value) { comment_ = value; }

  void ParameterSetDescription::setAllowAnything() { anythingAllowed_ = true; }

  void ParameterSetDescription::setUnknown() { unknown_ = true; }

  ParameterDescriptionNode* ParameterSetDescription::addNode(ParameterDescriptionNode const& node) {
    std::unique_ptr<ParameterDescriptionNode> clonedNode(node.clone());
    return addNode(std::move(clonedNode), false, true);
  }

  ParameterDescriptionNode* ParameterSetDescription::addNode(std::unique_ptr<ParameterDescriptionNode> node) {
    return addNode(std::move(node), false, true);
  }

  ParameterDescriptionNode* ParameterSetDescription::addOptionalNode(ParameterDescriptionNode const& node,
                                                                     bool writeToCfi) {
    std::unique_ptr<ParameterDescriptionNode> clonedNode(node.clone());
    return addNode(std::move(clonedNode), true, writeToCfi);
  }

  ParameterDescriptionNode* ParameterSetDescription::addOptionalNode(std::unique_ptr<ParameterDescriptionNode> node,
                                                                     bool writeToCfi) {
    return addNode(std::move(node), true, writeToCfi);
  }

  ParameterDescriptionNode* ParameterSetDescription::addNode(std::unique_ptr<ParameterDescriptionNode> node,
                                                             bool optional,
                                                             bool writeToCfi) {
    std::set<std::string> nodeLabels;
    std::set<ParameterTypes> nodeParameterTypes;
    std::set<ParameterTypes> nodeWildcardTypes;
    node->checkAndGetLabelsAndTypes(nodeLabels, nodeParameterTypes, nodeWildcardTypes);
    throwIfLabelsAlreadyUsed(nodeLabels);
    throwIfWildcardCollision(nodeParameterTypes, nodeWildcardTypes);

    SetDescriptionEntry entry;
    entry.setOptional(optional);
    entry.setWriteToCfi(writeToCfi);
    entries_.push_back(entry);
    return entries_.back().setNode(std::move(node));
  }

  void ParameterSetDescription::validate(ParameterSet& pset) const {
    using std::placeholders::_1;
    if (unknown_)
      return;

    std::set<std::string> validatedLabels;
    for_all(entries_, std::bind(&ParameterSetDescription::validateNode, _1, std::ref(pset), std::ref(validatedLabels)));

    std::vector<std::string> parameterNames = pset.getParameterNames();
    if (validatedLabels.size() != parameterNames.size()) {
      // Three labels will be magically inserted into the top level
      // of a module ParameterSet even though they are not in the
      // python configuration files.  If these are present, then
      // assume they are OK and count them as validated.

      std::string module_label("@module_label");
      if (pset.exists(module_label)) {
        validatedLabels.insert(module_label);
      }

      std::string module_type("@module_type");
      if (pset.exists(module_type)) {
        validatedLabels.insert(module_type);
      }

      std::string module_edm_type("@module_edm_type");
      if (pset.exists(module_edm_type)) {
        validatedLabels.insert(module_edm_type);
      }

      std::string service_type("@service_type");
      if (pset.exists(service_type)) {
        validatedLabels.insert(service_type);
      }

      // Try again
      if (validatedLabels.size() != parameterNames.size()) {
        if (IllegalParameters::throwAnException() && !anythingAllowed()) {
          throwIllegalParameters(parameterNames, validatedLabels);
        }
      }
    }
  }

  void ParameterSetDescription::writeCfi(std::ostream& os, bool startWithComma, int indentation) const {
    using std::placeholders::_1;
    bool wroteSomething = false;

    for_all(entries_,
            std::bind(&ParameterSetDescription::writeNode,
                      _1,
                      std::ref(os),
                      std::ref(startWithComma),
                      indentation,
                      std::ref(wroteSomething)));

    if (wroteSomething) {
      char oldFill = os.fill();
      os << "\n" << std::setfill(' ') << std::setw(indentation - 2) << "" << std::setfill(oldFill);
    }
  }

  void ParameterSetDescription::validateNode(SetDescriptionEntry const& entry,
                                             ParameterSet& pset,
                                             std::set<std::string>& validatedLabels) {
    entry.node()->validate(pset, validatedLabels, entry.optional());
  }

  void ParameterSetDescription::print(std::ostream& os, DocFormatHelper& dfh) const {
    using std::placeholders::_1;
    if (isUnknown()) {
      dfh.indent(os);
      os << "Description is unknown.  The configured PSet will not be validated\n";
      dfh.indent(os);
      os << "because the plugin has not defined this parameter set description.\n";
      if (!dfh.brief())
        os << "\n";
    }

    if (anythingAllowed()) {
      dfh.indent(os);
      os << "Description allows anything. If the configured PSet contains illegal parameters,\n";
      dfh.indent(os);
      os << "then validation will ignore them instead of throwing an exception.\n";
      if (!dfh.brief())
        os << "\n";
    }

    if (entries_.empty()) {
      dfh.indent(os);
      os << "Description is empty\n";
      if (!dfh.brief())
        os << "\n";
      return;
    }

    // Zeroth pass is only to calculate column widths in advance of any printing
    dfh.setPass(0);
    dfh.setCounter(0);
    for_all(entries_, std::bind(&ParameterSetDescription::printNode, _1, std::ref(os), std::ref(dfh)));

    // First pass prints top level parameters and references to structure
    dfh.setPass(1);
    dfh.setCounter(0);
    for_all(entries_, std::bind(&ParameterSetDescription::printNode, _1, std::ref(os), std::ref(dfh)));

    // Second pass prints substructure that goes into different sections of the
    // output document
    dfh.setPass(2);
    dfh.setCounter(0);
    for_all(entries_, std::bind(&ParameterSetDescription::printNode, _1, std::ref(os), std::ref(dfh)));
  }

  bool ParameterSetDescription::isLabelUnused(std::string const& label) const {
    return usedLabels_.find(label) == usedLabels_.end();
  }

  void ParameterSetDescription::throwIllegalParameters(std::vector<std::string> const& parameterNames,
                                                       std::set<std::string> const& validatedLabels) {
    std::set<std::string> parNames(parameterNames.begin(), parameterNames.end());

    std::set<std::string> diffNames;
    std::insert_iterator<std::set<std::string> > insertIter(diffNames, diffNames.begin());
    std::set_difference(parNames.begin(), parNames.end(), validatedLabels.begin(), validatedLabels.end(), insertIter);

    std::stringstream ss;
    for (std::set<std::string>::const_iterator iter = diffNames.begin(), iEnd = diffNames.end(); iter != iEnd; ++iter) {
      ss << " '" << *iter << "'\n";
    }
    if (diffNames.size() == 1U) {
      throw edm::Exception(errors::Configuration)
          << "Illegal parameter found in configuration.  The parameter is named:\n"
          << ss.str() << "You could be trying to use a parameter name that is not\n"
          << "allowed for this plugin or it could be misspelled.\n";
    } else {
      throw edm::Exception(errors::Configuration)
          << "Illegal parameters found in configuration.  The parameters are named:\n"
          << ss.str() << "You could be trying to use parameter names that are not\n"
          << "allowed for this plugin or they could be misspelled.\n";
    }
  }

  void ParameterSetDescription::writeNode(
      SetDescriptionEntry const& entry, std::ostream& os, bool& startWithComma, int indentation, bool& wroteSomething) {
    if (entry.writeToCfi()) {
      entry.node()->writeCfi(os, entry.optional(), startWithComma, indentation, wroteSomething);
    }
  }

  void ParameterSetDescription::printNode(SetDescriptionEntry const& entry, std::ostream& os, DocFormatHelper& dfh) {
    if (dfh.pass() < 2) {
      entry.node()->print(os, entry.optional(), entry.writeToCfi(), dfh);
    } else {
      entry.node()->printNestedContent(os, entry.optional(), dfh);
    }
  }

  void ParameterSetDescription::throwIfLabelsAlreadyUsed(std::set<std::string> const& nodeLabels) {
    std::set<std::string> duplicateLabels;
    std::insert_iterator<std::set<std::string> > insertIter(duplicateLabels, duplicateLabels.begin());
    std::set_intersection(nodeLabels.begin(), nodeLabels.end(), usedLabels_.begin(), usedLabels_.end(), insertIter);
    if (duplicateLabels.empty()) {
      usedLabels_.insert(nodeLabels.begin(), nodeLabels.end());
    } else {
      std::stringstream ss;
      for (std::set<std::string>::const_iterator iter = duplicateLabels.begin(), iEnd = duplicateLabels.end();
           iter != iEnd;
           ++iter) {
        ss << " \"" << *iter << "\"\n";
      }
      throw edm::Exception(errors::LogicError) << "Labels used in different nodes of a ParameterSetDescription\n"
                                               << "must be unique.  The following duplicate labels were detected:\n"
                                               << ss.str() << "\n";
    }
  }

  void ParameterSetDescription::throwIfWildcardCollision(std::set<ParameterTypes> const& nodeParameterTypes,
                                                         std::set<ParameterTypes> const& nodeWildcardTypes) {
    // 1. Check that the new wildcard types do not collide with the existing
    // parameter types.
    // 2. Check that the new parameter types do not collide with the existing
    // wildcard types.
    // 3. Then insert them.
    // The order of those steps is important because a wildcard with a default
    // value could insert a type in both sets and this is OK.

    // We assume the node already checked for collisions between the new parameter
    // types and the new wildcard types before passing the sets to this function.

    if (!nodeWildcardTypes.empty()) {
      std::set<ParameterTypes> duplicateTypes1;
      std::insert_iterator<std::set<ParameterTypes> > insertIter1(duplicateTypes1, duplicateTypes1.begin());
      std::set_intersection(typesUsedForParameters_.begin(),
                            typesUsedForParameters_.end(),
                            nodeWildcardTypes.begin(),
                            nodeWildcardTypes.end(),
                            insertIter1);

      if (!duplicateTypes1.empty()) {
        std::stringstream ss;
        for (std::set<ParameterTypes>::const_iterator iter = duplicateTypes1.begin(), iEnd = duplicateTypes1.end();
             iter != iEnd;
             ++iter) {
          ss << " \"" << parameterTypeEnumToString(*iter) << "\"\n";
        }
        throw edm::Exception(errors::LogicError)
            << "Within a ParameterSetDescription, the type used for a wildcard must\n"
            << "not be the same as the type used for other parameters. This rule\n"
            << "is violated for the following types:\n"
            << ss.str() << "\n";
      }
    }

    if (!typesUsedForWildcards_.empty()) {
      std::set<ParameterTypes> duplicateTypes2;
      std::insert_iterator<std::set<ParameterTypes> > insertIter2(duplicateTypes2, duplicateTypes2.begin());
      std::set_intersection(typesUsedForWildcards_.begin(),
                            typesUsedForWildcards_.end(),
                            nodeParameterTypes.begin(),
                            nodeParameterTypes.end(),
                            insertIter2);

      if (!duplicateTypes2.empty()) {
        std::stringstream ss;
        for (std::set<ParameterTypes>::const_iterator iter = duplicateTypes2.begin(), iEnd = duplicateTypes2.end();
             iter != iEnd;
             ++iter) {
          ss << " \"" << parameterTypeEnumToString(*iter) << "\"\n";
        }
        throw edm::Exception(errors::LogicError)
            << "Within a ParameterSetDescription, the type used for a wildcard must\n"
            << "not be the same as the type used for other parameters. This rule is\n"
            << "violated for the following types :\n"
            << ss.str() << "\n";
      }
    }

    typesUsedForParameters_.insert(nodeParameterTypes.begin(), nodeParameterTypes.end());
    typesUsedForWildcards_.insert(nodeWildcardTypes.begin(), nodeWildcardTypes.end());
  }

  ParameterDescriptionNode* ParameterSetDescription::ifExists(ParameterDescriptionNode const& node1,
                                                              ParameterDescriptionNode const& node2,
                                                              bool optional,
                                                              bool writeToCfi) {
    std::unique_ptr<ParameterDescriptionNode> pdIfExists = std::make_unique<IfExistsDescription>(node1, node2);
    return addNode(std::move(pdIfExists), optional, writeToCfi);
  }
}  // namespace edm
