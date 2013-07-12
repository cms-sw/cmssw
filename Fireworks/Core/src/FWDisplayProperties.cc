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
// $Id: FWDisplayProperties.cc,v 1.11 2010/06/17 14:02:37 matevz Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWDisplayProperties.h"
#include "Fireworks/Core/interface/FWColorManager.h"

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
