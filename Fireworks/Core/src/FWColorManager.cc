// -*- C++ -*-
//
// Package:     Core
// Class  :     FWColorManager
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Mar 24 10:10:01 CET 2009
//

// system include files
#include <iostream>
#include <map>
#include <boost/shared_ptr.hpp>
#include "TColor.h"
#include "TROOT.h"
#include "TMath.h"
#include "TEveUtil.h"
#include "TEveManager.h"
#include "TGLViewer.h"

// user include files
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/src/fwPaletteClassic.cc"
#include "Fireworks/Core/src/fwPaletteExtra.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//
const Color_t FWColorManager::s_defaultStartColorIndex    = 1000;
Color_t FWColorManager::getDefaultStartColorIndex()    { return s_defaultStartColorIndex; }

static
void resetColors(const float(* iColors)[3], unsigned int iSize, unsigned int iStart,  float gammaOff )
{
   // std::cout << "reset colors " << iColors << " start " << iStart << " size " << iSize<< " gamma " << gammaOff << std::endl;
   TSeqCollection* colorTable = gROOT->GetListOfColors();
   
   TColor* c = static_cast<TColor*>(colorTable->At(iStart));
   unsigned int index = iStart;
   if(0==c || c->GetNumber() != static_cast<int>(iStart)) {
      TIter   next(colorTable);
      while( (c=static_cast<TColor*>( next() )) ) {
         if(c->GetNumber()==static_cast<int>(iStart)) {
            index = iStart;
            break;
         }
      }
   }
   assert(0!=c);
   
   for(unsigned int i = index; i< index+iSize; ++i,++iColors) {
      TColor* c = static_cast<TColor*> (colorTable->At(i));
      float red = (*iColors)[0];
      float green = (*iColors)[1];
      float blue = (*iColors)[2];
     
      // apply brightness
      red     = TMath::Power(red,   (2.5 + gammaOff)/2.5);
      green   = TMath::Power(green, (2.5 + gammaOff)/2.5);
      blue    = TMath::Power(blue,  (2.5 + gammaOff)/2.5);

      // printf("--> [%d] (%.1f, %.1f, %.1f) => \n", i,  red, green, blue);
      c->SetRGB(red,green,blue);
   }
}
//
// constructors and destructor
//
FWColorManager::FWColorManager(FWModelChangeManager* iManager):
   m_paletteId(kClassic),
   m_gammaOff(0),
   m_background(kBlack),
   m_foreground(kWhite),
   m_changeManager(iManager),
   m_startColorIndex(0),
   m_numColorIndices(0),
   m_geomTransparency2D(50),
   m_geomTransparency3D(90)
{
   setDefaultGeomColors();
}

FWColorManager::~FWColorManager()
{
}

//
//
// member functions

void FWColorManager::setDefaultGeomColors()
{
   m_geomColor[kFWPixelBarrelColorIndex   ] = 1032;
   m_geomColor[kFWPixelEndcapColorIndex   ] = 1033;

   m_geomColor[kFWTrackerBarrelColorIndex  ] = 1026;
   m_geomColor[kFWTrackerEndcapColorIndex  ] = 1017;

   m_geomColor[kFWMuonBarrelLineColorIndex] = 1025;
   m_geomColor[kFWMuonEndcapLineColorIndex] = 1022;

   switch (m_paletteId) {
      case (kArctic):
         // m_geomColor[kFWMuonBarrelLineColorIndex] = 1027;
         //m_geomColor[kFWMuonEndcapLineColorIndex] = 1027;
         break;
      case (kFall):
         m_geomColor[kFWMuonBarrelLineColorIndex] = 1030;
         break;
      case (kSpring):
         m_geomColor[kFWMuonBarrelLineColorIndex] = 1032;
         m_geomColor[kFWMuonEndcapLineColorIndex] = 1032;
         break;
      case (kPurple):
         m_geomColor[kFWMuonBarrelLineColorIndex] = 1027; //kBlue -1;
         m_geomColor[kFWMuonEndcapLineColorIndex] = 1027;
         break;
      default:
         break;
   }
}

void FWColorManager::initialize()
{
  m_startColorIndex = s_defaultStartColorIndex;
  m_numColorIndices = fireworks::s_size;

  int index = m_startColorIndex;
  //std::cout <<"start color index "<<m_startColorIndex<<std::endl;
   
  const float(* itEnd)[3] = fireworks::s_forBlack+fireworks::s_size;
  for(const float(* it)[3] = fireworks::s_forBlack;
      it != itEnd;
      ++it) {
    //NOTE: this constructor automatically places this color into the gROOT color list
    //std::cout <<" color "<< index <<" "<<(*it)[0]<<" "<<(*it)[1]<<" "<<(*it)[2]<<std::endl;
    if ( index <= gROOT->GetListOfColors()->GetLast())
      gROOT->GetListOfColors()->RemoveAt(index);
    new TColor(index++,(*it)[0],(*it)[1],(*it)[2]);
  }

  // Save default ROOT colors.
  TEveUtil::SetColorBrightness(0, kFALSE);
}

void FWColorManager::setPalette(long long x)
{
   FWChangeSentry sentry(*m_changeManager);
   m_paletteId = (EPalette)x;
   setDefaultGeomColors();
   initColorTable();
}




void
FWColorManager::initColorTable()
{ 
   const float(* colValues)[3];
   colValues = isColorSetLight() ? fireworks::s_forWhite : fireworks::s_forBlack;
   if (m_paletteId == EPalette::kClassic)
   {
      // std::cout << "initColorTable classic \n";
      resetColors(colValues, fireworks::s_size, m_startColorIndex, m_gammaOff);
   }
   else {
      // std::cout << "initColorTable extra \n";
      float (*ev)[3] = (float (*)[3])calloc(3*fireworks::s_size, sizeof (float));
      for (int ci = 0; ci < 34; ++ci) {
         for (int j = 0; j < 3; ++j) 
            ev[ci][j] = colValues[ci][j];
        
      }
      fireworks::GetColorValuesForPaletteExtra(ev, fireworks::s_size, m_paletteId, isColorSetLight());
      resetColors(ev, fireworks::s_size, m_startColorIndex, m_gammaOff);
   }

   // AMT: Commented out ... Why this is necessary ?
   //float eveGamma =  isColorSetLight() ? 1.666*m_gammaOff - 2.5 : 1.666*m_gammaOff;
   //TEveUtil::SetColorBrightness(eveGamma);
}

void FWColorManager::updateColors()
{
   initColorTable();
   propagatePaletteChanges();
}

void FWColorManager::propagatePaletteChanges() const
{
   FWChangeSentry sentry(*m_changeManager);
   geomColorsHaveChanged_();
   colorsHaveChanged_();
   colorsHaveChangedFinished_(); 
}

void
FWColorManager::setBrightness(int b)
{
   // Called from CmsShowBrightnessPopup slider where range is set
   // to: -15, 15.
   m_gammaOff = -b*0.1f;
   updateColors();
}

int
FWColorManager::brightness()
{
  return TMath::FloorNint(-m_gammaOff*10);
}

void
FWColorManager::defaultBrightness()
{
   m_gammaOff = 0;
   updateColors();
}

void 
FWColorManager::switchBackground()
{ 
   setBackgroundColorIndex(isColorSetDark() ? kWhiteIndex : kBlackIndex);
}

void 
FWColorManager::setBackgroundColorIndex(BackgroundColorIndex iIndex)
{
   if(backgroundColorIndex()!=iIndex) {
      if(backgroundColorIndex()==kBlackIndex) {
         m_background=kWhiteIndex;
         m_foreground=kBlackIndex;
      } else {
         m_background=kBlackIndex;
         m_foreground=kWhiteIndex;
      }
      updateColors();
   }
}

void 
FWColorManager::setBackgroundAndBrightness(BackgroundColorIndex iIndex, int b)
{
   m_gammaOff = -b*0.1f;
   printf("set brightnes %f\n", m_gammaOff);
   setBackgroundColorIndex(iIndex);
}

Bool_t 
FWColorManager::setColorSetViewer(TGLViewer* v, Color_t iColor)
{
  if ( (iColor == kBlackIndex && !v->IsColorSetDark()) ||
       (iColor == kWhiteIndex && v->IsColorSetDark()) )
   { 
      v->SwitchColorSet();
      return kTRUE;
   }
   return kFALSE;
}

void
FWColorManager::setGeomColor(FWGeomColorIndex idx, Color_t iColor)
{
   // printf("set geom color %d \n", iColor);
   m_geomColor[idx] = iColor;
   geomColorsHaveChanged_();
   gEve->Redraw3D();
}
void
FWColorManager::setGeomTransparency(Color_t iTransp, bool projectedType)
{
   if (projectedType)
      m_geomTransparency2D = iTransp;
   else
      m_geomTransparency3D = iTransp;

   geomTransparencyHaveChanged_.emit(projectedType);

   gEve->Redraw3D();
}

//
// const member functions
//


void
FWColorManager::fillLimitedColors(std::vector<Color_t>& cv) const
{
   cv.reserve(cv.size() + m_numColorIndices);
   for (Color_t i = m_startColorIndex; i < borderOfLimitedColors(); ++i)
   {
      cv.push_back(i);
   }
}

FWColorManager::BackgroundColorIndex 
FWColorManager::backgroundColorIndex() const
{
   if(m_background==kBlack) {
      return kBlackIndex;
   }
   return kWhiteIndex;
}

bool 
FWColorManager::colorHasIndex(Color_t iColor) const
{
   return iColor > 0 && iColor < m_startColorIndex + m_numColorIndices;
}


Color_t 
FWColorManager::geomColor(FWGeomColorIndex iIndex) const
{
   return m_geomColor[iIndex];
}


static boost::shared_ptr<std::map<Color_t,Color_t> > m_oldColorToIndexMap;

Color_t
FWColorManager::oldColorToIndex(Color_t iColor, int version) const
{
   enum {
      kFWRed     = 1008,
      kFWBlue    = 1005,
      kFWCyan    = 1007,
      kFWGreen   = 1009,
      kFWMagenta = 1001,
      kFWOrange  = 1004,
      kFWYellow  = 1000
   };

   if (version < 3)
   {
      if(0==m_oldColorToIndexMap.get()) {
         m_oldColorToIndexMap = boost::shared_ptr<std::map<Color_t,Color_t> >(new std::map<Color_t,Color_t>());
         (*m_oldColorToIndexMap)[kRed]=kFWRed;
         (*m_oldColorToIndexMap)[kBlue]=kFWBlue;
         (*m_oldColorToIndexMap)[kYellow]=kFWYellow;
         (*m_oldColorToIndexMap)[kGreen]=kFWGreen;
         (*m_oldColorToIndexMap)[kCyan]=kFWCyan;
         (*m_oldColorToIndexMap)[kTeal]=kFWCyan;
         (*m_oldColorToIndexMap)[kMagenta]=kFWMagenta;
         (*m_oldColorToIndexMap)[kViolet]=kFWMagenta;
         (*m_oldColorToIndexMap)[kOrange]=kFWOrange;
         (*m_oldColorToIndexMap)[3]=kFWGreen;
      
      }
      return (*m_oldColorToIndexMap)[iColor];
   }
   else if (version == 3)
   {
      return iColor+ 1000;
   }
   else
   {
      const static unsigned int s_version45offset = 5;
      return iColor < 1011 ? iColor : iColor + s_version45offset ;
   }
}
