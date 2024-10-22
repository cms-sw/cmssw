#ifndef FWCore_ParameterSet_allowedValues_h
#define FWCore_ParameterSet_allowedValues_h
// -*- C++ -*-
//
// Package:     ParameterSet
// Function:    allowedValues
//
/**\function allowedValues allowedValues.h FWCore/ParameterSet/interface/allowedValues.h

 Description: Used to describe the allowed values in a ParameterSet

 Usage:
      The function is used in conjunction with ParameterSetDescription::ifValue to just constrain the allowed values
 and not add additional dependent ParameterSet nodes (which is allowed by ifValue).
 \code
   edm::ParameterSetDescription psetDesc;
   psetDesc.ifValue(edm::ParameterDescription<std::string>("sswitch", "a", true),
                    edm::allowedValues<std::string>("a", "h", "z") );
\endcode

 Implementation Details:
    The node associated with each allowed value is the EmptyGroupDescription which is just a dummy place holder.

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Jul 28 15:18:40 EDT 2020
//

#include "FWCore/ParameterSet/interface/ParameterDescriptionCases.h"
#include "FWCore/ParameterSet/interface/EmptyGroupDescription.h"

#include <memory>

namespace edm {

  template <typename T, typename... ARGS>
  std::unique_ptr<edm::ParameterDescriptionCases<T>> allowedValues(ARGS&&... args) {
    return (... || (std::forward<ARGS>(args) >> edm::EmptyGroupDescription()));
  }

}  // namespace edm

#endif
