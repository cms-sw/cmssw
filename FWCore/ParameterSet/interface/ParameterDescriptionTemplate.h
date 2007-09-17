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
// $Id$
//

// system include files

// user include files
#include "FWCore/ParameterSet/interface/ParameterDescription.h"

// forward declarations
namespace edm {
  template<class T>
class ParameterDescriptionTemplate : public ParameterDescription
{

   public:
   ParameterDescriptionTemplate(const std::string& iLabel,
                                bool isTracked):
  ParameterDescription(iLabel, isTracked, ParameterTypeToEnum::toEnum<T>()) {}

      // ---------- const member functions ---------------------
      virtual void validate(const ParameterSet&) const { return; }

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      ParameterDescriptionTemplate(const ParameterDescriptionTemplate&); // stop default

      const ParameterDescriptionTemplate& operator=(const ParameterDescriptionTemplate&); // stop default

      // ---------- member data --------------------------------

};

}
#endif
