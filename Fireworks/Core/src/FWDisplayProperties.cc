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
// $Id: FWDisplayProperties.cc,v 1.7 2010/06/03 13:38:32 eulisse Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWDisplayProperties.h"
#include "TColor.h"
#include "TROOT.h"

// A static default property.
const FWDisplayProperties FWDisplayProperties::defaultProperties(kWhite, true, 100);

FWDisplayProperties::FWDisplayProperties(const Color_t& iColor,
                                         bool isVisible,
                                         Int_t transparency) 
   : m_isVisible(isVisible),
     m_transparency(transparency)
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
