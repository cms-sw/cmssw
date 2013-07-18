#ifndef Fireworks_Core_FWViewContext_h
#define Fireworks_Core_FWViewContext_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWViewContext
// 
/**\class FWViewContext FWViewContext.h Fireworks/Core/interface/FWViewContext.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Wed Apr 14 18:31:27 CEST 2010
// $Id: FWViewContext.h,v 1.5 2010/09/26 19:54:56 amraktad Exp $
//

// system include files
#include <sigc++/sigc++.h>
#include <map>
#include <string>
#include "Rtypes.h"

// user include files

// forward declarations
class FWViewEnergyScale;

class FWViewContext
{
public:
   FWViewContext();
   virtual ~FWViewContext();

   FWViewEnergyScale* getEnergyScale() const;
   void setEnergyScale(FWViewEnergyScale*);

   void scaleChanged();

   mutable sigc::signal<void, const FWViewContext*> scaleChanged_;
   
private:
   FWViewContext(const FWViewContext&); // stop default

   const FWViewContext& operator=(const FWViewContext&); // stop default

   // ---------- member data --------------------------------

   FWViewEnergyScale* m_energyScale;
};


#endif
