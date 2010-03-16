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
// $Id: FWColorManager.h,v 1.17 2009/10/08 17:44:40 amraktad Exp $
//

// system include files
#include <vector>
#include "sigc++/signal.h"
#include "Rtypes.h"

// user include files

// forward declarations
class FWModelChangeManager;
class TGLViewer;

enum FWGeomColorIndex {
   kFWMuonBarrelMainColorIndex,
   kFWMuonBarrelLineColorIndex,
   kFWMuonEndCapMainColorIndex,
   kFWMuonEndCapLineColorIndex,
   kFWTrackerColorIndex,
   kFWCalo3DFrameColorIndex,
   kFWLegoFrameColorIndex,
   kFWLegoBoundraryColorIndex,
   kFWLegoFontColorIndex
};

class FWColorManager {

public:
   FWColorManager(FWModelChangeManager*);
   virtual ~FWColorManager();
   
   void initialize(Bool_t limit_palette);

   // ---------- const member functions ---------------------
   Color_t background() const {return m_background;}
   Color_t foreground() const {return m_foreground;}
   Bool_t  isColorSetDark() const {return m_background == kBlackIndex;}
   Bool_t  isColorSetLight() const {return m_background == kWhiteIndex;}
 
   Bool_t  hasLimitedPalette() const {return m_limitPalette;}

   Color_t indexToColor(unsigned int) const;
   unsigned int numberOfIndicies() const;
   
   unsigned int colorToIndex(Color_t) const;

   //help with backward compatibility with old config files
   unsigned int oldColorToIndex(Color_t) const;
   
   bool colorHasIndex(Color_t) const;
   
   Color_t geomColor(FWGeomColorIndex) const;
   
   enum BackgroundColorIndex { kWhiteIndex = kWhite, kBlackIndex = kBlack };
   BackgroundColorIndex backgroundColorIndex() const;
   // ---------- static member functions --------------------
   
   static Bool_t setColorSetViewer(TGLViewer*, Color_t);
   // ---------- member functions ---------------------------
   void defaultBrightness();
   void setBrightness(int);
   int  brightness ();
   void setBackgroundColorIndex(BackgroundColorIndex);
   void setBackgroundAndBrightness(BackgroundColorIndex, int);
   void switchBackground();

   mutable sigc::signal<void> colorsHaveChanged_;
   //called after all the slots attached to colorsHaveChanged_ are done
   mutable sigc::signal<void> colorsHaveChangedFinished_;

private:
   FWColorManager(const FWColorManager&); // stop default
   
   const FWColorManager& operator=(const FWColorManager&); // stop default
   void updateColors();

   // ---------- member data --------------------------------

   Float_t m_gammaOff;

   Color_t m_background;
   Color_t m_foreground;
   FWModelChangeManager* m_changeManager;
   
   unsigned int m_startColorIndex;
   unsigned int m_numColorIndices;
   unsigned int m_startGeomColorIndex;

   Bool_t m_limitPalette;
};


#endif
