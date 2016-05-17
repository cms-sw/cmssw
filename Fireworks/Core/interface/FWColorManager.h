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
//

// system include files
#include <vector>
#include "sigc++/signal.h"
#include "Rtypes.h"

// user include files

// forward declarations
class FWModelChangeManager;
class TGLViewer;

enum FWGeomColorIndex
{
   kFWPixelBarrelColorIndex,
   kFWPixelEndcapColorIndex,
   kFWTrackerBarrelColorIndex,
   kFWTrackerEndcapColorIndex,
   kFWMuonBarrelLineColorIndex,
   kFWMuonEndcapLineColorIndex,
   kFWGeomColorSize
};


class FWColorManager
{
public:
   enum EPalette { kClassic, kArctic, kFall, kSpring, kPurple };

   FWColorManager(FWModelChangeManager*);
   virtual ~FWColorManager();
   
   void initialize();

   // ---------- const member functions ---------------------
   Color_t background() const {return m_background;}
   Color_t foreground() const {return m_foreground;}
   Bool_t  isColorSetDark() const {return m_background == kBlackIndex;}
   Bool_t  isColorSetLight() const {return m_background == kWhiteIndex;}
 
   int numberOfLimitedColors() const {return m_numColorIndices;}
   int offsetOfLimitedColors() const {return m_startColorIndex;}
   int borderOfLimitedColors() const {return m_startColorIndex + m_numColorIndices;}

   void fillLimitedColors(std::vector<Color_t>& cv) const;

   //help with backward compatibility with old config files
   Color_t oldColorToIndex(Color_t, int version) const;
   
   bool colorHasIndex(Color_t) const;
   
   Color_t geomColor(FWGeomColorIndex) const;
   
   enum BackgroundColorIndex { kWhiteIndex = kWhite, kBlackIndex = kBlack };

   BackgroundColorIndex backgroundColorIndex() const;
   EPalette getPalette() const { return m_paletteId; }

   // ---------- static member functions --------------------
   
   static Bool_t setColorSetViewer(TGLViewer*, Color_t);

   static Color_t getDefaultStartColorIndex();

   // ---------- member functions ---------------------------

   void defaultBrightness();
   void setBrightness(int);
   int  brightness ();
   void setBackgroundColorIndex(BackgroundColorIndex);
   void setBackgroundAndBrightness(BackgroundColorIndex, int);
   void switchBackground();
   
   void setPalette(long long);

   void setGeomColor(FWGeomColorIndex, Color_t);
   void setGeomTransparency(Color_t idx, bool projectedType);
   Color_t geomTransparency(bool projected) const { return projected ? m_geomTransparency2D : m_geomTransparency3D; } 

   void setDefaultGeomColors();
   void propagatePaletteChanges() const;
   mutable sigc::signal<void> colorsHaveChanged_;
   mutable sigc::signal<void> geomColorsHaveChanged_;
   mutable sigc::signal<void, bool> geomTransparencyHaveChanged_;

   //called after all the slots attached to colorsHaveChanged_ are done
   mutable sigc::signal<void> colorsHaveChangedFinished_;

private:
   FWColorManager(const FWColorManager&); // stop default
   
   const FWColorManager& operator=(const FWColorManager&); // stop default
   void updateColors();
   void initColorTable();

   // ---------- member data --------------------------------
   EPalette m_paletteId;

   Float_t m_gammaOff;

   Color_t m_background;
   Color_t m_foreground;
   FWModelChangeManager* m_changeManager;
   
   Color_t m_startColorIndex;
   Color_t m_numColorIndices;

   Color_t m_geomColor[kFWGeomColorSize];
   Char_t  m_geomTransparency2D;
   Char_t  m_geomTransparency3D;

   static const Color_t s_defaultStartColorIndex;
};


#endif
