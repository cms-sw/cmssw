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
// Original Author:  
//         Created:  Wed Apr 14 18:31:27 CEST 2010
// $Id: FWViewContext.h,v 1.1 2010/04/15 20:15:15 amraktad Exp $
//

// system include files
#include <sigc++/sigc++.h>

// user include files

// forward declarations

class FWViewContext
{

public:
   FWViewContext();
   virtual ~FWViewContext();

   // ---------- const member functions ---------------------
   float getEnergyScale() const;
   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void  setEnergyScale(float);
   mutable sigc::signal<void, const FWViewContext*> scaleChanged_;
   
private:
   FWViewContext(const FWViewContext&); // stop default

   const FWViewContext& operator=(const FWViewContext&); // stop default

   // ---------- member data --------------------------------

   float m_energyScale;
};


#endif
