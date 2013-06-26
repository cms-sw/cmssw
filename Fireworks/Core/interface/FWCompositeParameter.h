#ifndef Fireworks_Core_FWCompositeParameter_h
#define Fireworks_Core_FWCompositeParameter_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWCompositeParameter
//
/**\class FWCompositeParameter FWCompositeParameter.h Fireworks/Core/interface/FWCompositeParameter.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Fri Mar  7 14:37:04 EST 2008
// $Id: FWCompositeParameter.h,v 1.3 2009/01/23 21:35:41 amraktad Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWParameterBase.h"
#include "Fireworks/Core/interface/FWParameterizable.h"

// forward declarations

class FWCompositeParameter : public FWParameterBase, public FWParameterizable
{

public:
   FWCompositeParameter(FWParameterizable* iParent,
                        const std::string& iName,
                        unsigned int iVersion=1);
   virtual ~FWCompositeParameter();

   // ---------- const member functions ---------------------
   virtual void addTo(FWConfiguration& ) const ;

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   virtual void setFrom(const FWConfiguration&);

private:
   FWCompositeParameter(const FWCompositeParameter&);    // stop default

   const FWCompositeParameter& operator=(const FWCompositeParameter&);    // stop default

   // ---------- member data --------------------------------
   unsigned int m_version;
};


#endif
