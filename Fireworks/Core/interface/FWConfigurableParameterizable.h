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
   ~FWConfigurableParameterizable() override;

   // ---------- const member functions ---------------------
   void addTo(FWConfiguration&) const override;

   unsigned int version() const {
      return m_version;
   }
   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void setFrom(const FWConfiguration&) override;

private:
   FWConfigurableParameterizable(const FWConfigurableParameterizable&) = delete;    // stop default

   const FWConfigurableParameterizable& operator=(const FWConfigurableParameterizable&) = delete;    // stop default

   // ---------- member data --------------------------------
   unsigned int m_version;
};


#endif
