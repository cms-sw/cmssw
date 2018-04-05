#ifndef Fireworks_Core_CmsShowCommon_h
#define Fireworks_Core_CmsShowCommon_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     CmsShowCommon
// 
/**\class CmsShowCommon CmsShowCommon.h Fireworks/Core/interface/CmsShowCommon.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Fri Sep 10 14:51:07 CEST 2010
//

#include <sigc++/signal.h>
#include <sigc++/sigc++.h>

#include "Rtypes.h"
#include "TGLUtil.h"
#include "TEveVector.h"

#include "Fireworks/Core/interface/FWConfigurableParameterizable.h"
#include "Fireworks/Core/interface/FWBoolParameter.h"
#include "Fireworks/Core/interface/FWLongParameter.h"
#include "Fireworks/Core/interface/FWEnumParameter.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWViewContext.h"
#include "Fireworks/Core/interface/FWBeamSpot.h"

class CmsShowCommonPopup;
class FWViewEnergyScale;
class FWColorManager;
namespace fireworks
{
class Context;
}

class CmsShowCommon : public FWConfigurableParameterizable
{
   friend class CmsShowCommonPopup;

public:
   CmsShowCommon(fireworks::Context*);
   ~CmsShowCommon() override;

   // ---------- const member functions ---------------------
   void addTo(FWConfiguration&) const override;

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void setFrom(const FWConfiguration&) override;

   void setTrackBreakMode();
   void setDrawBreakMarkers();

   int  gamma() { return m_gamma.value(); }
   void setGamma();
   void switchBackground();
   void permuteColors();
   void randomizeColors();
   void loopPalettes();

   void setGeomColor(FWGeomColorIndex, Color_t);
   void setGeomTransparency(int val, bool projected);
   FWViewEnergyScale* getEnergyScale() const { return m_viewContext.getEnergyScale(); }

  
   const TGLColorSet& getLightColorSet() const { return m_lightColorSet; }
   const TGLColorSet& getDarkColorSet()  const { return m_darkColorSet;  }

   
   UChar_t getProjTrackBreaking() const { return m_trackBreak.value(); }
   bool    getRnrPTBMarkers() const { return m_drawBreakPoints.value(); }

   void setView(CmsShowCommonPopup* x) { m_view= x;}
  
   void                 getEventCenter(float* inC) const;
   void                 setEventCenter(float, float, float);
   void                 resetEventCenter();
  
   mutable sigc::signal<void, const CmsShowCommon*> eventCenterChanged_;

protected:
   const FWColorManager*   colorManager() const;
   void setPalette();
   // ---------- member data --------------------------------

 
   CmsShowCommonPopup*        m_view;
   fireworks::Context*        m_context;

   FWEnumParameter            m_trackBreak;
   FWBoolParameter            m_drawBreakPoints;

   // general colors
   mutable FWLongParameter   m_backgroundColor; // can be set via Ctr+b key binding
   FWLongParameter           m_gamma;
   mutable FWEnumParameter   m_palette;

   // geom colors
   FWLongParameter     m_geomTransparency2D;
   FWLongParameter     m_geomTransparency3D;
   FWLongParameter*    m_geomColors[kFWGeomColorSize];

   TGLColorSet         m_lightColorSet;
   TGLColorSet         m_darkColorSet;
 
   FWViewContext        m_viewContext;

   bool                  m_useBeamSpot;
   TEveVector            m_externalEventCenter; //cached

private:
   CmsShowCommon(const CmsShowCommon&) = delete; // stop default
   const CmsShowCommon& operator=(const CmsShowCommon&) = delete; // stop default

};


#endif
