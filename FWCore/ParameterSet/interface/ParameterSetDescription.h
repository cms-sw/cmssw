#ifndef FWCore_ParameterSet_ParameterSetDescription_h
#define FWCore_ParameterSet_ParameterSetDescription_h
// -*- C++ -*-
//
// Package:     ParameterSet
// Class  :     ParameterSetDescription
// 
/**\class ParameterSetDescription ParameterSetDescription.h FWCore/ParameterSet/interface/ParameterSetDescription.h

 Description: Used to describe the allowed values in a ParameterSet

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Jul 31 15:18:40 EDT 2007
// $Id: ParameterSetDescription.h,v 1.7 2009/01/09 20:55:25 wmtan Exp $
//

#include "FWCore/ParameterSet/interface/ParameterDescription.h"
#include "FWCore/Utilities/interface/value_ptr.h"

#include <vector>
#include <string>
#include <memory>
#include <iosfwd>

namespace edm {

  class ParameterSet;

  class ParameterSetDescription
  {

  public:
    typedef std::vector<edm::value_ptr<ParameterDescription> > Parameters;
    typedef Parameters::const_iterator parameter_const_iterator;
        
    ParameterSetDescription();
    virtual ~ParameterSetDescription();

    ///allow any parameter label/value pairs
    void setAllowAnything();
      
    // This is set only for parameterizables which have not set their descriptions.
    // This should only be called to allow backwards compatibility.
    void setUnknown();

    template<class T, class U>
    ParameterDescription* add(U const& iLabel, T const& value) {
      return add<T, U>(iLabel, value, true, false);
    }

    template<class T, class U>
    ParameterDescription* addUntracked(U const& iLabel, T const& value) {
      return add<T, U>(iLabel, value, false, false);
    }

    template<class T, class U>
    ParameterDescription* addOptional(U const& iLabel, T const& value) {
      return add<T, U>(iLabel, value, true, true);
    }

    template<class T, class U>
    ParameterDescription* addOptionalUntracked(U const& iLabel, T const& value) {
      return add<T, U>(iLabel, value, false, true);
    }

    //Throws a cms::Exception if invalid
    void validate(ParameterSet const& pset) const;

    bool anythingAllowed() const { return anythingAllowed_; }
    bool isUnknown() const { return unknown_; }

    parameter_const_iterator parameter_begin() const {
      return parameters_.begin();
    }

    parameter_const_iterator parameter_end() const {
      return parameters_.end();
    }

    // Better performance if space is reserved for the number of
    // top level parameters before any are added.
    void reserve(Parameters::size_type n) {
      parameters_.reserve(n);
    }

    void writeCfi(std::ostream & os, bool startWithComma, int indentation) const; 

  private:

    template<class T, class U>
    ParameterDescription* add(U const& iLabel, T const& value, bool isTracked, bool isOptional);

    static void
    validateDescription(value_ptr<ParameterDescription> const& description,
                        ParameterSet const& pset);

    void
    validateName(std::string const& parameterName,
                 ParameterSet const& pset) const;

    static void
    match(value_ptr<ParameterDescription> const& description,
          std::string const& parameterName,
          ParameterSet const& pset,
          bool & foundMatch);

    static void
    throwIllegalParameter(std::string const& parameterName,
                          ParameterSet const& pset);

    static void writeParameter(value_ptr<ParameterDescription> const& description,
                               std::ostream & os,
                               bool  & startWithComma,
                               int indentation);

    bool anythingAllowed_;
    bool unknown_;
    Parameters parameters_;
  };
}

#include "FWCore/ParameterSet/interface/ParameterDescriptionTemplate.h"

namespace edm {

  template<class T, class U>
  ParameterDescription*
  ParameterSetDescription::
    add(U const& iLabel, T const& value, bool isTracked, bool isOptional) {

    std::auto_ptr<ParameterDescription> ptr(new ParameterDescriptionTemplate<T>(iLabel, isTracked, isOptional, value));

    edm::value_ptr<ParameterDescription> vptr;
    parameters_.push_back(vptr);
    parameters_.back() = ptr;

    return parameters_.back().operator->();
  }
}

#endif
