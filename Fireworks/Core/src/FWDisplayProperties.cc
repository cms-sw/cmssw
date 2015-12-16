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
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWDisplayProperties.h"
#include "Fireworks/Core/interface/FWColorManager.h"

// A static default property.
const FWDisplayProperties FWDisplayProperties::defaultProperties
(FWColorManager::getDefaultStartColorIndex(), true, true, 0);

FWDisplayProperties::FWDisplayProperties(Color_t iColor,
                                         bool    isVisible,
                                         bool    filterPassed,
                                         Char_t  transparency) 
   : m_color(iColor),
     m_isVisible(isVisible),
     m_filterPassed(filterPassed),
     m_transparency(transparency)
{}
