#ifndef Fireworks_ParticleFlow_FWPFScale_h
#define Fireworks_ParticleFlow_FWPFScale_h
// -*- C++ -*-
//
// Package:     ParticleFlow
// Class  :     FWPFScale
// 
/**\class FWPFScale FWPFScale.h Fireworks/ParticleFlow/interface/FWPFScale.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Fri Jun 18 17:01:24 CEST 2010
// $Id: FWPFScale.h,v 1.1 2010/06/18 19:51:25 amraktad Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWViewEnergyScale.h"


class FWPFScale : public FWViewEnergyScale
{
public:
   FWPFScale();
   virtual ~FWPFScale();

   // ---------- const member functions ---------------------
   virtual float getVal() const;

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

   virtual void reset(); 
   virtual void setVal(float);

   private:
   FWPFScale(const FWPFScale&); // stop default
   const FWPFScale& operator=(const FWPFScale&); // stop default

   // ---------- member data --------------------------------

};



#endif
