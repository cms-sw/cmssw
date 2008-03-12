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
// $Id: ParameterDescription.h,v 1.1 2007/09/17 21:04:37 chrjones Exp $
//

// system include files
#include <string>

// user include files

// forward declarations
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

namespace edm {
  enum ParameterTypes {
    k_uint32,    k_vuint32,
    k_int32,     k_vint32,
    k_uint64,    k_vuint64,
    k_int64,     k_vint64,
    k_string,    k_vstring,
    k_bool,      k_vbool,
    k_double,    k_vdouble,
    k_PSet,      k_VPSet,
    k_FileInPath,
    k_InputTag,  k_VInputTag,
    k_EventID,   k_VEventID,
    k_LuminosityBlockID,   k_VLuminosityBlockID,
    k_numParameterTypes
  };
  std::string parameterTypeEnumToString(ParameterTypes);
  
    struct ParameterTypeToEnum {
      template <class T>
      static ParameterTypes toEnum();
    };
  
class ParameterDescription
{

   public:
      virtual ~ParameterDescription();

      // ---------- const member functions ---------------------
      virtual void validate(const ParameterSet&) const = 0;
  
      const std::string& label() const {
        return label_;
      }

      ParameterTypes type() const {
        return type_;
      }
      
      bool isTracked() const {
        return isTracked_;
      }
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

  protected:
      ParameterDescription(const std::string& iLabel,
                           bool isTracked,
                           ParameterTypes iType);

   private:
      ParameterDescription(const ParameterDescription&); // stop default

      const ParameterDescription& operator=(const ParameterDescription&); // stop default

      // ---------- member data --------------------------------
      std::string label_;
      ParameterTypes type_;
      bool isTracked_;
};

}
#endif
