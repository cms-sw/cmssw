// -*- C++ -*-
//
// Package:     ParticleFlow
// Class  :     FWPFScale
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  
//         Created:  Fri Jun 18 19:06:47 CEST 2010
// $Id: FWPFScale.cc,v 1.1 2010/06/18 19:51:25 amraktad Exp $
//

// system include files

// user include files
#include "Fireworks/ParticleFlow/src/FWPFScale.h"
#include "TMath.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWPFScale::FWPFScale():
   FWViewEnergyScale(0.f)
{
}

// FWPFScale::FWPFScale(const FWPFScale& rhs)
// {
//    // do actual copying here;
// }

FWPFScale::~FWPFScale()
{
}

void 
FWPFScale::setVal(float s)
{
   if (s > m_value)
      m_value = s;
}

float 
FWPFScale::getVal() const
{
   if (m_value > 0.f)
      return 1.f/m_value;
   else 
      return 1.f;
}

void
FWPFScale::reset()
{
   m_value = 0; 
}
