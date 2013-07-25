#ifndef Fireworks_Core_FWEnumParameter_h
#define Fireworks_Core_FWEnumParameter_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWEnumParameter
// 
/**\class FWEnumParameter FWEnumParameter.h Fireworks/Core/interface/FWEnumParameter.h

 Description: Specialization of FWLongParameter to allow drop-down menu GUI.

 Usage:
    <usage>

*/
//
// Original Author:  matevz
//         Created:  Fri Apr 30 15:16:55 CEST 2010
// $Id: FWEnumParameter.h,v 1.1 2010/04/30 15:29:44 matevz Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWLongParameter.h"
#include <map>

// forward declarations

class FWEnumParameter : public FWLongParameter
{

   public:
   FWEnumParameter() : FWLongParameter()
   {}

   FWEnumParameter(FWParameterizable* iParent,
                   const std::string& iName,
                   const long &iDefault=0,
                   long iMin=-1,
                   long iMax=-1) :
      FWLongParameter(iParent, iName, iDefault, iMin, iMax)
   {}

   template <class K>
   FWEnumParameter(FWParameterizable* iParent,
                   const std::string& iName,
                   K iCallback,
                   const long &iDefault=0,
                   long iMin=-1,
                   long iMax=-1) :
      FWLongParameter(iParent, iName, iCallback, iDefault, iMin, iMax)
   {}

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

   bool addEntry(Long_t id, const std::string& txt)
   {
      return m_enumEntries.insert(std::make_pair(id, txt)).second;
   }

   const std::map<Long_t, std::string>& entryMap() const { return m_enumEntries; }

private:
   FWEnumParameter(const FWEnumParameter&);                  // stop default
   const FWEnumParameter& operator=(const FWEnumParameter&); // stop default

   // ---------- member data --------------------------------
   std::map<Long_t, std::string> m_enumEntries;
};

#endif
