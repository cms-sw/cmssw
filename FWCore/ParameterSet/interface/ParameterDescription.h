#ifndef FWCore_ParameterSet_ParameterDescription_h
#define FWCore_ParameterSet_ParameterDescription_h
// -*- C++ -*-
//
// Package:     ParameterSet
// Class  :     ParameterDescription
// 
/**\class ParameterDescription ParameterDescription.h FWCore/ParameterSet/interface/ParameterDescription.h

 Description: Base class for a description of one parameter in a ParameterSet

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Aug  2 15:33:46 EDT 2007
// $Id: ParameterDescription.h,v 1.3 2008/11/14 19:41:22 wdd Exp $
//

#include <boost/shared_ptr.hpp>
#include <string>
#include <vector>

namespace edm {

  class ParameterSet;
  class ParameterSetDescription;

  // The values of this enumeration match the values
  // defined in the ParameterSet Entry class, to make
  // comparisons easier.
  enum ParameterTypes {
    k_int32 = 'I',
    k_vint32 = 'i',
    k_uint32 = 'U',
    k_vuint32 = 'u',
    k_int64 = 'L',
    k_vint64 = 'l',
    k_uint64 = 'X',
    k_vuint64 = 'x',
    k_double = 'D',
    k_vdouble = 'd',
    k_bool = 'B',
    k_vbool = 'b',
    k_string = 'S',
    k_vstring = 's',
    k_EventID = 'E',
    k_VEventID = 'e',
    k_LuminosityBlockID = 'M',
    k_VLuminosityBlockID = 'm',
    k_InputTag = 't',
    k_VInputTag = 'v',
    k_FileInPath = 'F',
    k_PSet = 'P',
    k_VPSet = 'p'
  };

  std::string parameterTypeEnumToString(ParameterTypes iType);

  struct ParameterTypeToEnum {
    template <class T>
    static ParameterTypes toEnum();
  };

  class ParameterDescription
  {
  public:
    virtual ~ParameterDescription();

    void validate(const ParameterSet& pset) const;

    const std::string& label() const { return label_; }
    ParameterTypes type() const { return type_; }
    bool isTracked() const { return isTracked_; }
    bool optional() const { return optional_; }

    // Need to define something like this that returns the value in a
    // string although we probably want to format the string in python
    // for use in python configurations
    // void defaultValue(std::string& value) const { defaultValue_(value); }

    boost::shared_ptr<ParameterSetDescription> parameterSetDescription()
      { return parameterSetDescription_; }

    boost::shared_ptr<std::vector<ParameterSetDescription> > parameterSetDescriptions()
      { return parameterSetDescriptions_; }

    void setParameterSetDescription(boost::shared_ptr<ParameterSetDescription> const& value)
      { parameterSetDescription_ = value; }

    void setParameterSetDescriptions(boost::shared_ptr<std::vector<ParameterSetDescription> > const& value)
      { parameterSetDescriptions_ = value; }

    void throwParameterNotDefined() const;

  protected:
    ParameterDescription(const std::string& iLabel,
                         bool isTracked,
                         bool optional,
                         ParameterTypes iType);

  private:
    ParameterDescription(const ParameterDescription&); // stop default
    const ParameterDescription& operator=(const ParameterDescription&); // stop default

    virtual void validate_(const ParameterSet& pset, bool & exists) const = 0;
    void validateParameterSetDescription(const ParameterSet& pset) const;
    void validateParameterSetDescriptions(const ParameterSet& pset) const;
    // virtual void defaultValue_(std::string& value) const = 0;

    std::string label_;
    ParameterTypes type_;
    bool isTracked_;
    bool optional_;

    boost::shared_ptr<ParameterSetDescription> parameterSetDescription_;
    boost::shared_ptr<std::vector<ParameterSetDescription> > parameterSetDescriptions_;
  };
}
#endif
