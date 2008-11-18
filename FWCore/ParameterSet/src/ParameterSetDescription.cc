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
// $Id: ParameterSetDescription.cc,v 1.5 2008/11/18 15:10:39 wdd Exp $
//

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

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
  ParameterSetDescription::validate(const edm::ParameterSet& pset) const
  {
    if (unknown_ || anythingAllowed()) return;

    for (parameter_const_iterator pdesc = parameter_begin(),
                                  pend = parameter_end();
         pdesc != pend;
         ++pdesc) {
      (*pdesc)->validate(pset);
    }

    std::vector<std::string> psetNames = pset.getParameterNames();
    for (std::vector<std::string>::const_iterator iter = psetNames.begin(),
	                                          iEnd = psetNames.end();
         iter != iEnd;
         ++iter) {
      if (*iter == std::string("@module_label") ||
          *iter == std::string("@module_type")) continue;

      std::string const& parameterName = *iter;

      bool foundMatchingParameter = false;
      for (parameter_const_iterator pdesc = parameter_begin(),
                                    pend = parameter_end();
           pdesc != pend;
           ++pdesc) {
        if (parameterName == (*pdesc)->label()) {
          Entry const* entry = pset.retrieveUnknown(parameterName);
          if (entry->typeCode() == (*pdesc)->type() &&
              entry->isTracked() == (*pdesc)->isTracked()) {
            foundMatchingParameter = true;
          }
        }
      }
      if (!foundMatchingParameter) {
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
  }

  template<>
  boost::shared_ptr<ParameterDescription>
  ParameterSetDescription::add<ParameterSetDescription>(const std::string& iLabel,
                                                        ParameterSetDescription const& value,
                                                        bool isTracked,
                                                        bool optional) {
    boost::shared_ptr<ParameterDescription> ptr(new ParameterDescriptionTemplate<ParameterSet>(iLabel, isTracked, optional, ParameterSet()));
    boost::shared_ptr<ParameterSetDescription> copyOfSet(new ParameterSetDescription(value));
    ptr->setParameterSetDescription(copyOfSet);
    parameters_.push_back(ptr);
    return ptr;
  }

  template<>
  boost::shared_ptr<ParameterDescription>
  ParameterSetDescription::add<std::vector<ParameterSetDescription> >(const std::string& iLabel,
                                                                      std::vector<ParameterSetDescription> const& value,
                                                                      bool isTracked,
                                                                      bool optional) {
    boost::shared_ptr<ParameterDescription> ptr(new ParameterDescriptionTemplate<std::vector<ParameterSet> >(iLabel, isTracked, optional, std::vector<ParameterSet>()));
    boost::shared_ptr<std::vector<ParameterSetDescription> > copyOfSet(new std::vector<ParameterSetDescription>(value));
    ptr->setParameterSetDescriptions(copyOfSet);
    parameters_.push_back(ptr);
    return ptr;
  }
}
