#ifndef Fireworks_Core_FWItemValueGetter_h
#define Fireworks_Core_FWItemValueGetter_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWItemValueGetter
//
/**\class FWItemValueGetter FWItemValueGetter.h Fireworks/Core/interface/FWItemValueGetter.h

   Description: Retrieves a particular value from an item

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Sun Nov 30 16:14:58 EST 2008
// $Id: FWItemValueGetter.h,v 1.1 2008/12/01 00:51:21 chrjones Exp $
//

// system include files
#include <string>
#include <vector>
#include "Reflex/Member.h"
#include "Reflex/Type.h"
// user include files

// forward declarations

class FWItemValueGetter {

public:
   FWItemValueGetter(const ROOT::Reflex::Type&,
                     const std::vector<std::pair<std::string, std::string> >& iFindValueFrom);
   //virtual ~FWItemValueGetter();

   // ---------- const member functions ---------------------
   double valueFor(const void*) const;
   std::string stringValueFor(const void*) const;

   bool isValid() const;

   std::string valueName() const;
   const std::string& unit() const;
   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void setValueAndUnit(const std::string& iValue, const std::string& iUnit);

private:
   //FWItemValueGetter(const FWItemValueGetter&); // stop default

   //const FWItemValueGetter& operator=(const FWItemValueGetter&); // stop default

   // ---------- member data --------------------------------
   ROOT::Reflex::Type m_type;
   ROOT::Reflex::Member m_memberFunction;
   std::string m_unit;
};


#endif
