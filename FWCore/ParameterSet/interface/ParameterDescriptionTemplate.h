#ifndef FWCore_ParameterSet_ParameterDescriptionTemplate_h
#define FWCore_ParameterSet_ParameterDescriptionTemplate_h
// -*- C++ -*-
//
// Package:     ParameterSet
// Class  :     ParameterDescriptionTemplate
// 
/**\class ParameterDescriptionTemplate ParameterDescriptionTemplate.h FWCore/ParameterSet/interface/ParameterDescriptionTemplate.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Aug  2 15:33:51 EDT 2007
// $Id: ParameterDescriptionTemplate.h,v 1.1 2007/09/17 21:04:37 chrjones Exp $
//

#include "FWCore/ParameterSet/interface/ParameterDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
// #include "FWCore/ParameterSet/interface/types.h"

#include <string>

namespace edm {

  template<class T>
  class ParameterDescriptionTemplate : public ParameterDescription {
  public:

    ParameterDescriptionTemplate(const std::string& iLabel,
                                 bool isTracked,
                                 T const& value):
      
      ParameterDescription(iLabel, isTracked, ParameterTypeToEnum::toEnum<T>()),
      value_(value) {
    }

  private:
    ParameterDescriptionTemplate(const ParameterDescriptionTemplate&); // stop default
    const ParameterDescriptionTemplate& operator=(const ParameterDescriptionTemplate&); // stop default

    virtual void validate_(const ParameterSet& pset) const {

      // See if pset has a parameter matching this ParameterDescription
      // In the future, the current plan is to have this insert missing
      // parameters into the ParameterSet with the correct default value.
      // Cannot do that until we get a non const ParameterSet passed in.
      if (!pset.existsAs<T>(label(), isTracked())) throwParameterNotDefined();
    }

    // virtual void defaultValue_(std::string& value) const {
    //   edm::encode(value, value_);
    // }

    // This holds the default value of the parameter, except
    // when the parameter is another ParameterSet or vector<ParameterSet>.
    // In those cases it just holds a default constructed ParameterSet or
    // empty vector which serves no purpose.
    T value_;
  };
}
#endif
