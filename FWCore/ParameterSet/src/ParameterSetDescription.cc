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
#include "FWCore/ParameterSet/interface/Entry.h"
#include "FWCore/Utilities/interface/Algorithms.h"

#include "boost/bind.hpp"

namespace edm {

  ParameterSetDescription::ParameterSetDescription():
  anythingAllowed_(false),
  unknown_(false) {
  }

  ParameterSetDescription::~ParameterSetDescription() {
  }

  void 
  ParameterSetDescription::setAllowAnything()
  {
    anythingAllowed_ = true;
  }

  void 
  ParameterSetDescription::setUnknown()
  {
    unknown_ = true;
  }

  void
  ParameterSetDescription::validate(ParameterSet const& pset) const
  {
    if (unknown_ || anythingAllowed()) return;

    for_all(parameters_,
            boost::bind(&ParameterSetDescription::validateDescription, _1, boost::cref(pset)));

    std::vector<std::string> parameterNames = pset.getParameterNames();
    for_all(parameterNames,
            boost::bind(&ParameterSetDescription::validateName, boost::cref(this), _1, boost::cref(pset)));
  }

  void ParameterSetDescription::
  validateDescription(value_ptr<ParameterDescription> const& description,
                      ParameterSet const& pset) {
    description->validate(pset);
  }

  void
  ParameterSetDescription::validateName(std::string const& parameterName,
                                        ParameterSet const& pset) const {

    if (parameterName == std::string("@module_label") ||
        parameterName == std::string("@module_type")) return;

    bool foundMatch = false;

    for_all(parameters_,
            boost::bind(&ParameterSetDescription::match,
                        _1,
                        boost::cref(parameterName),
                        boost::cref(pset),
                        boost::ref(foundMatch)));

    if (!foundMatch) throwIllegalParameter(parameterName, pset);
  }

  void ParameterSetDescription::
  match(value_ptr<ParameterDescription> const& description,
        std::string const& parameterName,
        ParameterSet const& pset,
        bool & foundMatch) {

    if (parameterName == description->label()) {
      Entry const* entry = pset.retrieveUnknown(parameterName);
      if (entry->typeCode() == description->type() &&
          entry->isTracked() == description->isTracked()) {
         foundMatch = true;
      }
    }
  }

  void
  ParameterSetDescription::throwIllegalParameter(std::string const& parameterName,
                                                 ParameterSet const& pset) {
    // prepare an error message and throw
    Entry const* entry = pset.retrieveUnknown(parameterName);

    std::string tr;
    if (entry->isTracked()) tr = std::string("as a tracked");
    else tr = std::string("as an untracked");

    ParameterTypes type = static_cast<ParameterTypes>(entry->typeCode());

    throw edm::Exception(errors::Configuration)
      << "Illegal parameter found in configuration.  It is named \"" 
      << parameterName << "\"\n"
      << "and defined " << tr 
      << " " << parameterTypeEnumToString(type) << ".\n"
      << "You could be trying to use a parameter name that is not\n"
      << "allowed for this module.  Or it could be mispelled, of\n"
      << "the wrong type, or incorrectly declared tracked or untracked.\n";
  }
}
