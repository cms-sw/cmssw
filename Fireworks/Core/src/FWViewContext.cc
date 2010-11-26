// -*- C++ -*-
//
// Package:     Core
// Class  :     FWViewContext
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Wed Apr 14 18:31:58 CEST 2010
// $Id: FWViewContext.cc,v 1.6 2010/09/26 19:57:21 amraktad Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWViewContext.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"

FWViewContext::FWViewContext()
{
}

FWViewContext::~FWViewContext()
{
}


void
FWViewContext::scaleChanged()
{
   scaleChanged_.emit(this);
}

FWViewEnergyScale*
FWViewContext::getEnergyScale() const
{
   return m_energyScale;
}

void
FWViewContext::setEnergyScale(FWViewEnergyScale* x)
{
   m_energyScale =x;
}
