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
// $Id: ParameterSetDescription.cc,v 1.3 2008/10/08 22:13:36 wmtan Exp $
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
        throw edm::Exception(errors::Configuration)
          << "Unexpected parameter \"" << parameterName << "\" was defined.  It could be a typo";
      }
    }
  }
}
