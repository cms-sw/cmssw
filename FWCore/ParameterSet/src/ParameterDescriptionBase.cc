// -*- C++ -*-
//
// Package:     ParameterSet
// Class  :     ParameterDescriptionBase
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Aug  2 15:35:43 EDT 2007
//

#include "FWCore/ParameterSet/interface/ParameterDescriptionBase.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <iostream>
#include <iomanip>

namespace edm {

  ParameterDescriptionBase::ParameterDescriptionBase(std::string const& iLabel,
                                                     ParameterTypes iType,
                                                     bool isTracked)
    :label_(iLabel),
     type_(iType),
     isTracked_(isTracked)
  { }

  ParameterDescriptionBase::ParameterDescriptionBase(char const* iLabel,
                                                     ParameterTypes iType,
                                                     bool isTracked)
    :label_(iLabel),
     type_(iType),
     isTracked_(isTracked)
  { }

  ParameterDescriptionBase::~ParameterDescriptionBase() { }

  void
  ParameterDescriptionBase::throwParameterWrongTrackiness() const {
    std::string tr("a tracked");
    std::string shouldBe("untracked");
    if (isTracked()) {
      tr = "an untracked";
      shouldBe = "tracked";
    }

    throw edm::Exception(errors::Configuration)
      << "In the configuration, parameter \"" << label() << "\" is defined "
      "as " << tr << " " << parameterTypeEnumToString(type()) << ".\n"
      << "It should be " << shouldBe << ".\n";
  }

  void
  ParameterDescriptionBase::throwParameterWrongType() const {
    std::string tr("an untracked");
    if (isTracked()) tr = "a tracked";

    throw edm::Exception(errors::Configuration)
      << "Parameter \"" << label() << "\" should be defined "
      "as " << tr << " " << parameterTypeEnumToString(type()) << ".\n"
      << "The type in the configuration is incorrect.\n";
  }

  void
  ParameterDescriptionBase::
  checkAndGetLabelsAndTypes_(std::set<std::string> & usedLabels,
                             std::set<ParameterTypes> & parameterTypes,
                             std::set<ParameterTypes> & wildcardTypes) const {
    usedLabels.insert(label());
    parameterTypes.insert(type());
  }

  void
  ParameterDescriptionBase::
  writeCfi_(std::ostream & os,
           bool & startWithComma,
           int indentation,
           bool & wroteSomething) const {

    wroteSomething = true;
    if (startWithComma) os << ",";
    startWithComma = true;

    os << "\n" << std::setfill(' ') << std::setw(indentation) << "";

    os << label()
       << " = cms.";
    if (!isTracked()) os << "untracked.";
    os << parameterTypeEnumToString(type())
       << "(";
    writeCfi_(os, indentation);
    os << ")";
  }

  bool
  ParameterDescriptionBase::
  partiallyExists_(ParameterSet const& pset) const {
    return exists(pset);
  }

  int
  ParameterDescriptionBase::
  howManyExclusiveOrSubNodesExist_(ParameterSet const& pset) const {
    return exists(pset) ? 1 : 0; 
  }
}
