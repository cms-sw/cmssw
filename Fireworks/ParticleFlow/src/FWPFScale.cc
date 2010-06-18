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
// $Id$
//

// system include files

// user include files
#include "TMath.h"
#include "Fireworks/ParticleFlow/src/FWPFScale.h"


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
   m_et(0),
   m_pt(0)
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
   // after begin event search max et
   if (s > m_et) m_et = s;
}

float 
FWPFScale::getVal() const
{
   // lego height is by default TMath::Pi()
   // as it is scales in lego view

   if (m_et > 0.f)
      return TMath::Pi()/m_et;
   else 
      return TMath::Pi();
}

void
FWPFScale::reset()
{
   m_et = 0.f; 
   m_pt = 0; 
}
