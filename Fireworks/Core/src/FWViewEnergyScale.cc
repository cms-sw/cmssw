// -*- C++ -*-
//
// Package:     Core
// Class  :     FWViewEnergyScale
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Fri Jun 18 20:37:44 CEST 2010
// $Id$
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWViewEnergyScale.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWViewEnergyScale::FWViewEnergyScale():
   m_valid(false), m_value(1) 
{
}

FWViewEnergyScale::~FWViewEnergyScale()
{
}



void  
FWViewEnergyScale::setVal(float s)
{
   m_value = s;
}

void  
FWViewEnergyScale::reset()
{
   m_value = 1.f;
}

float  
FWViewEnergyScale::getVal() const
{
   return m_value;
}

