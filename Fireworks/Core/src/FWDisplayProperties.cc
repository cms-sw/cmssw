// -*- C++ -*-
//
// Package:     Core
// Class  :     FWDisplayProperties
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Thu Jan  3 17:05:44 EST 2008
// $Id: FWDisplayProperties.cc,v 1.4 2009/01/23 21:35:42 amraktad Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWDisplayProperties.h"
#include "TColor.h"
#include "TROOT.h"

// A static default property.
const FWDisplayProperties FWDisplayProperties::defaultProperties(kWhite, true);

FWDisplayProperties::FWDisplayProperties(const Color_t& iColor,
                                         bool isVisible) 
: m_isVisible(isVisible)
{
   setColor(iColor);
}

void 
FWDisplayProperties::setColor(Color_t iColor) 
{
   // make sure the color is availabe in ROOT
   // for colors above 100
   if ( !gROOT->GetColor(iColor) && iColor >= 100 ){
      m_color = TColor::GetColorDark(iColor-100);
      return;
   }
   m_color = iColor;
}
