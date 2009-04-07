#ifndef Fireworks_Core_FWColorManager_h
#define Fireworks_Core_FWColorManager_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWColorManager
// 
/**\class FWColorManager FWColorManager.h Fireworks/Core/interface/FWColorManager.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Mar 24 10:07:58 CET 2009
// $Id: FWColorManager.h,v 1.1 2009/04/07 13:55:55 chrjones Exp $
//

// system include files
#include <vector>
#include "sigc++/signal.h"
#include "Rtypes.h"

// user include files

// forward declarations
class FWModelChangeManager;

enum FWGeomColorIndex {
   kFWMuonBarrelMainColorIndex,
   kFWMuonBarrelLineColorIndex,
   kFWMuonEndCapMainColorIndex,
   kFWMuonEndCapLineColorIndex,
   kFWTrackerColorIndex
};

class FWColorManager {

public:
   FWColorManager(FWModelChangeManager*);
   virtual ~FWColorManager();
   
   // ---------- const member functions ---------------------
   Color_t background() const {return m_background;}
   Color_t foreground() const {return m_foreground;}
   
   Color_t indexToColor(unsigned int) const;
   unsigned int numberOfIndicies() const;
   
   unsigned int colorToIndex(Color_t) const;

   //help with backward compatibility with old config files
   unsigned int oldColorToIndex(Color_t) const;
   
   bool colorHasIndex(Color_t) const;
   
   Color_t geomColor(FWGeomColorIndex) const;
   
   enum BackgroundColorIndex {kBlackIndex, kWhiteIndex};
   BackgroundColorIndex backgroundColorIndex() const;
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   void setBackgroundColorIndex(BackgroundColorIndex);
   
   sigc::signal<void> colorsHaveChanged_;

private:
   FWColorManager(const FWColorManager&); // stop default
   
   const FWColorManager& operator=(const FWColorManager&); // stop default
   
   // ---------- member data --------------------------------
   Color_t m_background;
   Color_t m_foreground;
   FWModelChangeManager* m_changeManager;
   
   unsigned int m_startColorIndex;
   unsigned int m_startGeomColorIndex;
};


#endif
