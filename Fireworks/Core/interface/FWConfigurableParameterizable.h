#ifndef Fireworks_Core_FWConfigurableParameterizable_h
#define Fireworks_Core_FWConfigurableParameterizable_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWConfigurableParameterizable
//
/**\class FWConfigurableParameterizable FWConfigurableParameterizable.h Fireworks/Core/interface/FWConfigurableParameterizable.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Sun Mar 16 12:01:29 EDT 2008
// $Id: FWConfigurableParameterizable.h,v 1.3 2009/01/23 21:35:41 amraktad Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWParameterizable.h"
#include "Fireworks/Core/interface/FWConfigurable.h"


// forward declarations

class FWConfigurableParameterizable : public FWParameterizable, public FWConfigurable
{

public:
   FWConfigurableParameterizable(unsigned int iVersion = 1);
   virtual ~FWConfigurableParameterizable();

   // ---------- const member functions ---------------------
   virtual void addTo(FWConfiguration&) const;

   unsigned int version() const {
      return m_version;
   }
   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   virtual void setFrom(const FWConfiguration&);

private:
   FWConfigurableParameterizable(const FWConfigurableParameterizable&);    // stop default

   const FWConfigurableParameterizable& operator=(const FWConfigurableParameterizable&);    // stop default

   // ---------- member data --------------------------------
   unsigned int m_version;
};


#endif
