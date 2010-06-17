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
// $Id: FWDisplayProperties.cc,v 1.10 2010/06/16 14:04:40 matevz Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWDisplayProperties.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "TColor.h"
#include "TROOT.h"

// A static default property.
const FWDisplayProperties FWDisplayProperties::defaultProperties
(FWColorManager::getDefaultStartColorIndex(), true, 0);

FWDisplayProperties::FWDisplayProperties(Color_t iColor,
                                         bool    isVisible,
                                         Char_t  transparency) 
   : m_color(iColor),
     m_isVisible(isVisible),
     m_transparency(transparency)
{}
