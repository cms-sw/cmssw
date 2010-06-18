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
// $Id: FWViewContext.cc,v 1.2 2010/05/03 15:47:38 amraktad Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWViewContext.h"
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
FWViewContext::FWViewContext()
{
}

// FWViewContext::FWViewContext(const FWViewContext& rhs)
// {
//    // do actual copying here;
// }

FWViewContext::~FWViewContext()
{
   for (Scales_i i = m_scales.begin(); i != m_scales.end(); i++)
   {
      delete i->second;
   }
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
FWViewContext::scaleChanged()
{
   scaleChanged_.emit(this);
}

void
FWViewContext::resetScale()
{
   for (Scales_i i = m_scales.begin(); i != m_scales.end(); ++i)
   {
      i->second->reset();
   }
}

void 
FWViewContext::addScale(const std::string& name, FWViewEnergyScale* s) const
{
   m_scales[name] = s;
}

FWViewEnergyScale*
FWViewContext::getEnergyScale(const std::string& type) const
{
   Scales_i it = m_scales.find(type);
   if (it != m_scales.end())
   {
      return it->second;
   }
   return 0;
}
