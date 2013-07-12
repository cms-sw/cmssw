#ifndef Fireworks_Core_FWParameterBase_h
#define Fireworks_Core_FWParameterBase_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWParameterBase
//
/**\class FWParameterBase FWParameterBase.h Fireworks/Core/interface/FWParameterBase.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Sat Feb 23 13:35:15 EST 2008
// $Id: FWParameterBase.h,v 1.3.40.1 2012/02/18 01:58:26 matevz Exp $
//

// system include files
#include <string>

// user include files
#include "Fireworks/Core/interface/FWConfigurable.h"

// forward declarations
class FWConfiguration;
class FWParameterizable;

class FWParameterBase : public FWConfigurable
{
public:
   FWParameterBase(FWParameterizable* iParent, const std::string& iName);
   virtual ~FWParameterBase();

   // ---------- const member functions ---------------------

   //virtual void addTo(FWConfiguration& ) const = 0;
   const std::string& name() const { return m_name; }

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   //virtual void setFrom(const FWConfiguration&) = 0;

private:
   FWParameterBase(const FWParameterBase&);                  // stop default
   const FWParameterBase& operator=(const FWParameterBase&); // stop default

   // ---------- member data --------------------------------

   std::string m_name;
};

#endif
