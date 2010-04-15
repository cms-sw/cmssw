// -*- C++ -*-
//
// Package:     Core
// Class  :     FWViewContext
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  
//         Created:  Wed Apr 14 18:31:58 CEST 2010
// $Id$
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWViewContext.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWViewContext::FWViewContext():
m_energyScale(1.f)
{
}

// FWViewContext::FWViewContext(const FWViewContext& rhs)
// {
//    // do actual copying here;
// }

FWViewContext::~FWViewContext()
{
}

//
// assignment operators
//
// const FWViewContext& FWViewContext::operator=(const FWViewContext& rhs)
// {
//   //An exception safe implementation is
//   FWViewContext temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FWViewContext::setEnergyScale(float s)
{
   m_energyScale = s;
}

//
// const member functions
//
float
FWViewContext::getEnergyScale() const
{
   return  m_energyScale;
}
//
// static member functions
//
