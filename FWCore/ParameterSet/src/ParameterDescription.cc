// -*- C++ -*-
//
// Package:     ParameterSet
// Class  :     ParameterDescription
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Aug  2 15:35:43 EDT 2007
// $Id: ParameterDescription.cc,v 1.1 2007/09/17 21:04:38 chrjones Exp $
//

// system include files

// user include files
#include "FWCore/ParameterSet/interface/ParameterDescription.h"


//
// constants, enums and typedefs
//
#define TYPE_TO_NAME(type) case k_ ## type: return #type
#define TYPE_TO_ENUM(type,e_val) template<> ParameterTypes ParameterTypeToEnum::toEnum<type >(){ return e_val; }

namespace edm {

  TYPE_TO_ENUM(int,k_int32)
  TYPE_TO_ENUM(unsigned int,k_uint32)
  
  std::string parameterTypeEnumToString(ParameterTypes iType){
    switch(iType) {
      case k_uint32:
        return "uint32";
      case k_vuint32:
        return "vuint32";
      case k_int32:
        return "int32";
      case k_vint32:
        return "vint32";
      case k_uint64:
        return "uint64";
      case k_vuint64:
        return "vuint64";
      case k_int64:
        return "int64";
      case k_vint64:
        return "vint64";
      case k_string:
        return "string";
      case k_vstring:
        return "vstring";
      case k_bool:
        return "bool";
      case k_vbool:
        return "vbool";
        TYPE_TO_NAME(double);
        TYPE_TO_NAME(vdouble);
        TYPE_TO_NAME(PSet);
        TYPE_TO_NAME(VPSet);
        TYPE_TO_NAME(FileInPath);
        TYPE_TO_NAME(InputTag);
        TYPE_TO_NAME(VInputTag);
        TYPE_TO_NAME(EventID);
        TYPE_TO_NAME(VEventID);
        TYPE_TO_NAME(LuminosityBlockID);
        TYPE_TO_NAME(VLuminosityBlockID);
      default:
        assert(false);
    }
    return "";
  }
  
//
// static data member definitions
//

//
// constructors and destructor
//
  ParameterDescription::ParameterDescription(const std::string& iLabel,
                                             bool iIsTracked,
                                             ParameterTypes iType)
  :label_(iLabel),
  type_(iType),
  isTracked_(iIsTracked)
{
}

// ParameterDescription::ParameterDescription(const ParameterDescription& rhs)
// {
//    // do actual copying here;
// }

ParameterDescription::~ParameterDescription()
{
}

//
// assignment operators
//
// const ParameterDescription& ParameterDescription::operator=(const ParameterDescription& rhs)
// {
//   //An exception safe implementation is
//   ParameterDescription temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

//
// const member functions
//

//
// static member functions
//
}
