
// It is unusual to put this include before the header guard,
// but it guarantees the headers are included in the proper order.
// ParameterSetDescription.h must be included before 
// ParameterDescriptionTemplate.h
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

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
// $Id: ParameterDescriptionTemplate.h,v 1.5 2008/12/08 22:33:59 wdd Exp $
//

#include "FWCore/ParameterSet/interface/ParameterDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include <vector>

namespace edm {

  class ParameterSetDescription;

  template<class T>
  class ParameterDescriptionTemplate : public ParameterDescription {
  public:

    ParameterDescriptionTemplate(std::string const& iLabel,
                                 bool isTracked,
                                 bool isOptional,
                                 T const& value):
      // WARNING: the toEnum function is intentionally undefined if the template
      // parameter is ParameterSet or vector<ParameterSet>.  ParameterSetDescription
      // or vector<ParameterSetDescription> should be used instead.  This template
      // parameter is usually passed through from an add*<T> function of ParameterSetDescription.
      ParameterDescription(iLabel, ParameterTypeToEnum::toEnum<T>(), isTracked, isOptional),
      value_(value) {
    }

    ParameterDescriptionTemplate(char const* iLabel,
                                 bool isTracked,
                                 bool isOptional,
                                 T const& value):
      // WARNING: the toEnum function is intentionally undefined if the template
      // parameter is ParameterSet or vector<ParameterSet>.  ParameterSetDescription
      // or vector<ParameterSetDescription> should be used instead.  This template
      // parameter is usually passed through from an add*<T> function of ParameterSetDescription.
      ParameterDescription(iLabel, ParameterTypeToEnum::toEnum<T>(), isTracked, isOptional),
      value_(value) {
    }

    virtual ~ParameterDescriptionTemplate() { }

    virtual void validate(ParameterSet const& pset) const {

      bool exists = pset.existsAs<T>(label(), isTracked());

      // See if pset has a parameter matching this ParameterDescription
      // In the future, the current plan is to have this insert missing
      // parameters into the ParameterSet with the correct default value.
      // Cannot do that until we get a non const ParameterSet passed in.
      if (!isOptional() && !exists) throwParameterNotDefined();
    }

    virtual ParameterDescription* clone() const {
      return new ParameterDescriptionTemplate(*this);
    }

  private:

    // This holds the default value of the parameter, except
    // when the parameter is another ParameterSet or vector<ParameterSet>.
    // In those cases it holds the nested ParameterSetDescription or
    // vector<ParameterSetDescription>
    T value_;
  };

  template<>
  class ParameterDescriptionTemplate<ParameterSetDescription> : public ParameterDescription {

  public:

    ParameterDescriptionTemplate(std::string const& iLabel,
                                 bool isTracked,
                                 bool isOptional,
                                 ParameterSetDescription const& value);

    ParameterDescriptionTemplate(char const* iLabel,
                                 bool isTracked,
                                 bool isOptional,
                                 ParameterSetDescription const& value);

    virtual ~ParameterDescriptionTemplate();

    virtual void validate(ParameterSet const& pset) const;

    virtual ParameterSetDescription const* parameterSetDescription() const;
    virtual ParameterSetDescription * parameterSetDescription();

    virtual ParameterDescription* clone() const {
      return new ParameterDescriptionTemplate(*this);
    }

  private:
    ParameterSetDescription psetDesc_;
  };

  template<>
  class ParameterDescriptionTemplate<std::vector<ParameterSetDescription> > : public ParameterDescription {

  public:

    ParameterDescriptionTemplate(std::string const& iLabel,
                                 bool isTracked,
                                 bool isOptional,
                                 std::vector<ParameterSetDescription> const& vPsetDesc);

    ParameterDescriptionTemplate(char const* iLabel,
                                 bool isTracked,
                                 bool isOptional,
                                 std::vector<ParameterSetDescription> const& vPsetDesc);

    virtual ~ParameterDescriptionTemplate();

    virtual void validate(ParameterSet const& pset) const;

    virtual std::vector<ParameterSetDescription> const* parameterSetDescriptions() const;
    virtual std::vector<ParameterSetDescription> * parameterSetDescriptions();

    virtual ParameterDescription* clone() const {
      return new ParameterDescriptionTemplate(*this);
    }

  private:

    void
    validateDescription(ParameterSetDescription const& psetDescription,
                        std::vector<ParameterSet> const& psets,
                        int & i) const;

    std::vector<ParameterSetDescription> vPsetDesc_;
  };
}
#endif
