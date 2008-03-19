// -*- C++ -*-
//
// Package:     Core
// Class  :     accessMenuBar
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Mar 18 15:29:09 EDT 2008
// $Id$
//

// system include files
//EVIL, but avoids having to inherit from TEveBrowser just to access the menu
#define protected public
#include "TEveBrowser.h"
#undef protected

// user include files
#include "Fireworks/Core/src/accessMenuBar.h"



namespace fireworks {
   TGMenuBar* accessMenuBar(TEveBrowser* iBrowser)
   {
      return iBrowser->fMenuBar;
   }
}
