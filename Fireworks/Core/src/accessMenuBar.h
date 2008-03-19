#ifndef Fireworks_Core_accessMenuBar_h
#define Fireworks_Core_accessMenuBar_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     accessMenuBar
// 
/**\class accessMenuBar accessMenuBar.h Fireworks/Core/interface/accessMenuBar.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Mar 18 15:29:07 EDT 2008
// $Id$
//

// system include files

// user include files

// forward declarations
class TGMenuBar;
class TEveBrowser;

namespace fireworks {
   TGMenuBar* accessMenuBar(TEveBrowser*);
}

#endif
