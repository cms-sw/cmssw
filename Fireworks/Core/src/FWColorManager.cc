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
// $Id: FWColorManager.cc,v 1.27 2010/06/17 14:02:37 matevz Exp $
//

// system include files
#include <iostream>
#include <map>
#include <boost/shared_ptr.hpp>
#include "TColor.h"
#include "TROOT.h"
#include "TMath.h"
#include "TEveUtil.h"
#include "TGLViewer.h"

// user include files
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//
//static std::vector<Color_t>* s_forWhite=0;
//static std::vector<Color_t>* s_forBlack=0;

const Color_t FWColorManager::s_defaultStartColorIndex    = 1000;
const Color_t FWColorManager::s_defaultStartGeometryIndex = 1100;

Color_t FWColorManager::getDefaultStartColorIndex()    { return s_defaultStartColorIndex; }
Color_t FWColorManager::getDefaultStartGeometryIndex() { return s_defaultStartGeometryIndex; }

enum {
   kFWRed     = 1008,
   kFWBlue    = 1005,
   kFWCyan    = 1007,
   kFWGreen   = 1009,
   kFWMagenta = 1001,
   kFWOrange  = 1004,
   kFWYellow  = 1000
};

static const float s_forWhite[][3] ={
{ 0.79, 0.79, 0.12 }, //yellow (made it a bit darker)
{ 0.47, 0.00, 0.64 }, //purple
{ 0.98, 0.70, 0.00 }, //yellowish-orange
{ 0.18, 0.00, 0.59 }, //purplish-blue
{ 0.98, 0.54, 0.00 }, //orange
{ 0.00, 0.22, 1.00 }, //blue
{ 0.99, 0.26, 0.01 }, //dark orange
{ 0.00, 0.80, 0.78 }, //cyan
{ 1.00, 0.06, 0.00 }, //red
{ 0.33, 0.64, 0.14 }, //green
{ 0.60, 0.06, 0.23 }, //burgundy
{0.65, 0.92, 0.17}, //lime{ 0.99, 1.00, 0.39 },

{ 0.85, 0.85, 0.58 },
{ 0.87, 0.72, 0.92 },
{ 0.99, 0.88, 0.59 },
{ 0.79, 0.72, 0.90 },
{ 1.00, 0.82, 0.59 },
{ 0.71, 0.79, 0.99 },
{ 1.00, 0.80, 0.72 },
{ 0.71, 0.98, 0.95 },
{ 0.99, 0.74, 0.70 },
{ 0.77, 0.86, 0.65 },
{ 0.90, 0.74, 0.79 },
{ 0.67, 0.95, 0.52 }
};

static const float s_forBlack[][3] ={
{1.00, 1.00, 0.20}, //yellow
{0.53, 0.00, 0.69}, //purple
{0.98, 0.74, 0.01}, //yellowish-orange
{0.24, 0.00, 0.64}, //purplish-blue
{0.98, 0.60, 0.01}, //orange
{0.01, 0.28, 1.00}, //blue
{0.99, 0.33, 0.03}, //dark orange
{0.01, 0.83, 0.81}, //cyan
{1.00, 0.09, 0.00}, //red
{0.40, 0.69, 0.20}, //green
{0.65, 0.10, 0.29}, //burgundy
{0.65, 0.92, 0.17}, //lime
/*
{1.,0.,0.}, //red
{0.,0.,1.}, //blue
{0.,1.,1.}, //cyan
{0.,1.,0.}, //green
{1.,0.,1.}, //magenta
{1.,0.5,0.0},  //orange
{1.,1.,0.}, //yellow
{0.5,0.5,0.5}, //gray
*/
{0.27, 0.27, 0.04},
{0.19, 0.00, 0.24},
{0.19, 0.15, 0.00},
{0.14, 0.00, 0.38},
{0.19, 0.11, 0.00},
{0.01, 0.10, 0.33},
{0.17, 0.05, 0.02},
{0.00, 0.33, 0.29},
{0.34, 0.03, 0.01},
{0.15, 0.24, 0.06},
{0.24, 0.02, 0.11},
{0.22, 0.30, 0.07}
/*
{0.7,0.0,0.0},
{0.0,0.0,0.7},
{0.0,.7,0.7},
{0.0,.7,0.},
{.7,0.,.7},
{.7,0.4,0.0},
{.7,.7,0.0},
{0.3,0.3,0.3}
 */
};

static const unsigned int s_size = sizeof(s_forBlack)/sizeof(s_forBlack[0]);

static const float s_geomForWhite[][3] ={
{ 1.00, 0.44, 0.44 },
{ 1.00, 0.44, 0.44 },
{ 0.38, 0.63, 1.00 },
{ 0.14, 0.33, 1.00 },
{ 0.51, 1.00, 0.20 },
{ 0.77, 0.77, 0.77 }, // calo3d grid
{ 0.77, 0.77, 0.77 }, // lego grid
{ 0.54, 0.54, 0.54 }, // lego boundrary
{ 0.33, 0.33, 0.33 }  // lego font
};

static const float s_geomForBlack[][3] ={
{0x3f/256.,0.,0.},
{0x7f/256.,0.,0.},
{0.,0.,0x3f/256.},
{0.,0.,0x7f/256.},
{0.,0x7f/256.,0.},
{0.34, 0.34, 0.34}, // calo3d grid
{0.17, 0.17, 0.17},  // lego grid
{0.3, 0.3, 0.3}, // lego boundrary
{0.7, 0.7, 0.7}   // lego font
};

static const unsigned int s_geomSize = sizeof(s_geomForBlack)/sizeof(s_geomForBlack[0]);

//==============================================================================

static
void resetColors(const float(* iColors)[3], unsigned int iSize, unsigned int iStart,  float gammaOff )
{
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
      TColor* c = dynamic_cast<TColor*> (colorTable->At(i));
      float red = (*iColors)[0];
      float green = (*iColors)[1];
      float blue = (*iColors)[2];
     
      // apply brightness
      red     = TMath::Power(red,   (2.5 + gammaOff)/2.5);
      green   = TMath::Power(green, (2.5 + gammaOff)/2.5);
      blue    = TMath::Power(blue,  (2.5 + gammaOff)/2.5);

      c->SetRGB(red,green,blue);
   }
}
//
// constructors and destructor
//
FWColorManager::FWColorManager(FWModelChangeManager* iManager):
m_gammaOff(0),
m_background(kBlack),
m_foreground(kWhite),
m_changeManager(iManager),
m_numColorIndices(0)
{
}

// FWColorManager::FWColorManager(const FWColorManager& rhs)
// {
//    // do actual copying here;
// }

FWColorManager::~FWColorManager()
{
}

//
// assignment operators
//
// const FWColorManager& FWColorManager::operator=(const FWColorManager& rhs)
// {
//   //An exception safe implementation is
//   FWColorManager temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions

void FWColorManager::initialize()
{
  // Save default ROOT colors.
  TEveUtil::SetColorBrightness(0, kFALSE);

  m_startColorIndex = s_defaultStartColorIndex;
  m_numColorIndices = s_size;

  unsigned int index = m_startColorIndex;
  //std::cout <<"start color index "<<m_startColorIndex<<std::endl;
   
  const float(* itEnd)[3] = s_forBlack+s_size;
  for(const float(* it)[3] = s_forBlack;
      it != itEnd;
      ++it) {
    //NOTE: this constructor automatically places this color into the gROOT color list
    //std::cout <<" color "<< index <<" "<<(*it)[0]<<" "<<(*it)[1]<<" "<<(*it)[2]<<std::endl;
    new TColor(index++,(*it)[0],(*it)[1],(*it)[2]);
  }

  m_startGeomColorIndex = index = s_defaultStartGeometryIndex;
  itEnd = s_geomForBlack+s_geomSize;
  for(const float(* it)[3] = s_geomForBlack;
      it != itEnd;
      ++it) {
    //NOTE: this constructor automatically places this color into the gROOT color list
    new TColor(index++,(*it)[0],(*it)[1],(*it)[2]);
  }   
}

void FWColorManager::updateColors()
{
   if (backgroundColorIndex() == kBlackIndex)
   {
      resetColors(s_forBlack,s_size,m_startColorIndex,  m_gammaOff);
      resetColors(s_geomForBlack, s_geomSize, m_startGeomColorIndex,  m_gammaOff);
      TEveUtil::SetColorBrightness(1.666*m_gammaOff);
   }
   else
   {
      resetColors(s_forWhite,s_size,m_startColorIndex,  m_gammaOff);
      resetColors(s_geomForWhite, s_geomSize, m_startGeomColorIndex,  m_gammaOff);
      TEveUtil::SetColorBrightness(1.666*m_gammaOff - 2.5);
   }
   FWChangeSentry sentry(*m_changeManager);
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
   setBackgroundColorIndex(iIndex);
}

Bool_t 
FWColorManager::setColorSetViewer(TGLViewer* v, Color_t iColor)
{
  if ( (iColor == kBlackIndex && !v->IsColorSetDark()) ||
       (iColor == kWhiteIndex && v->IsColorSetDark()) )
   { 
      v->SwitchColorSet();
      v->RequestDraw(TGLRnrCtx::kLODHigh);

      return kTRUE;
   }
   return kFALSE;
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
   return m_startGeomColorIndex + iIndex;
}


static boost::shared_ptr<std::map<Color_t,Color_t> > m_oldColorToIndexMap;

Color_t
FWColorManager::oldColorToIndex(Color_t iColor) const
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
      //(*m_oldColorToIndexMap)[kGray]=kFWGray;
      (*m_oldColorToIndexMap)[3]=kFWGreen;
      
   }
   return (*m_oldColorToIndexMap)[iColor];
}

//
// static member functions
//
