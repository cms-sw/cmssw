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
// $Id: ParameterDescription.h,v 1.5.2.2 2008/12/14 16:38:31 wmtan Exp $
//

#include "FWCore/Utilities/interface/value_ptr.h"

#include <string>
#include <vector>

namespace edm {

  class ParameterSet;
  class ParameterSetDescription;

  // The values of this enumeration match the values
  // defined in the ParameterSet Entry class, to make
  // comparisons easier.
  // Exceptions are k_PSet and k_VPSet, which no longer
  // use the Entry class.
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
    k_string = 'S',
    k_vstring = 's',
    k_EventID = 'E',
    k_VEventID = 'e',
    k_LuminosityBlockID = 'M',
    k_VLuminosityBlockID = 'm',
    k_InputTag = 't',
    k_VInputTag = 'v',
    k_FileInPath = 'F',
    k_PSet = 'Q',
    k_VPSet = 'q'
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

    virtual void validate(ParameterSet const& pset) const = 0;

    std::string const& label() const { return label_; }
    ParameterTypes type() const { return type_; }
    bool isTracked() const { return isTracked_; }
    bool isOptional() const { return isOptional_; }

    virtual ParameterSetDescription const* parameterSetDescription() const { return 0; }
    virtual ParameterSetDescription * parameterSetDescription() { return 0; }

    virtual std::vector<ParameterSetDescription> const* parameterSetDescriptions() const { return 0; }
    virtual std::vector<ParameterSetDescription> * parameterSetDescriptions() { return 0; }

    void throwParameterNotDefined() const;

    virtual ParameterDescription* clone() const = 0;

  protected:
    ParameterDescription(std::string const& iLabel,
                         ParameterTypes iType,
                         bool isTracked,
                         bool isOptional
                         );

    ParameterDescription(char const* iLabel,
                         ParameterTypes iType,
                         bool isTracked,
                         bool isOptional
                         );
  private:
    std::string label_;
    ParameterTypes type_;
    bool isTracked_;
    bool isOptional_;
  };

  template <> 
  struct value_ptr_traits<ParameterDescription>  
  {
    static ParameterDescription * clone( ParameterDescription const * p ) { return p->clone(); }
  };
}
#endif
