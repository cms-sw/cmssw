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

 In addition to whatever you need to do to add a new type to the
 ParameterSet code, you need to do the following to add
 a new type to the ParameterSetDescription code:
 1.  add a value to the enumeration ParameterTypes (ParameterDescription.h)
 2.  add new TYPE_TO_NAME and TYPE_TO_ENUM macros (ParameterDescription.cc)
 3.  add declaration of writeValueToCfi function to ParameterDescriptionTemplate.h
 (Two of them if a vector of the type is also allowed)
 4.  define writeValueToCfi in ParameterDescriptionTemplate.cc
 5.  Consider whether you need a specialization of writeSingleValue and
 writeValueInVector in ParameterDescriptionTemplate.cc. The first is needed
 if operator<< for the new type does not print the correct format for a cfi.
 The second is needed if the format in a vector is different from the format
 when a single value is not in a vector.
 6.  add parameters of that type and vectors of that type to
 FWCore/Integration/test/ProducerWithPSetDesc.cc
 7.  Check and update the reference file in
 FWCore/Integration/test/unit_test_outputs/testProducerWithPsetDesc_cfi.py
*/
//
// Original Author:  Chris Jones
//         Created:  Thu Aug  2 15:33:46 EDT 2007
// $Id: ParameterDescription.h,v 1.8 2009/01/09 20:55:25 wmtan Exp $
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

    void writeCfi(std::ostream & os, int indentation) const;

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

    virtual void writeCfi_(std::ostream & os, int indentation) const = 0;

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
