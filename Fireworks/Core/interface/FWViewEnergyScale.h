#ifndef Fireworks_Core_FWViewEnergyScale_h
#define Fireworks_Core_FWViewEnergyScale_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWViewEnergyScale
// 
/**\class FWViewEnergyScale FWViewEnergyScale.h Fireworks/Core/interface/FWViewEnergyScale.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Fri Jun 18 20:37:55 CEST 2010
// $Id$
//

// system include files

// user include files

// forward declarations

class FWViewEnergyScale
{

public:
   FWViewEnergyScale();
   virtual ~FWViewEnergyScale();

   virtual void  setVal(float);
   virtual float getVal() const;
   virtual void reset();

private:
   FWViewEnergyScale(const FWViewEnergyScale&); // stop default

   const FWViewEnergyScale& operator=(const FWViewEnergyScale&); // stop default

   // ---------- member data --------------------------------
   bool  m_valid;
   float m_value;

};


#endif
