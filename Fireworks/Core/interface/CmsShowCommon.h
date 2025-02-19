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
// $Id: CmsShowCommon.h,v 1.11 2011/03/08 09:36:04 amraktad Exp $
//

#include <sigc++/signal.h>
#include <sigc++/sigc++.h>

#include "Rtypes.h"

#include "Fireworks/Core/interface/FWConfigurableParameterizable.h"
#include "Fireworks/Core/interface/FWBoolParameter.h"
#include "Fireworks/Core/interface/FWLongParameter.h"
#include "Fireworks/Core/interface/FWEnumParameter.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"

class CmsShowCommonPopup;
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
   virtual ~CmsShowCommon();

   // ---------- const member functions ---------------------
   virtual void addTo(FWConfiguration&) const;

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   virtual void setFrom(const FWConfiguration&);

   void setTrackBreakMode();
   void setDrawBreakMarkers();

   int  gamma() { return m_gamma.value(); }
   void setGamma();
   void switchBackground();

   void setGeomColor(FWGeomColorIndex, Color_t);
   void setGeomTransparency(int val, bool projected);

   FWViewEnergyScale* getEnergyScale() const { return m_energyScale.get(); }

   
   UChar_t getProjTrackBreaking() const { return m_trackBreak.value(); }
   bool    getRnrPTBMarkers() const { return m_drawBreakPoints.value(); }

protected:
   const FWColorManager*   colorManager() const;

   // ---------- member data --------------------------------

 
   fireworks::Context*        m_context;

   FWEnumParameter            m_trackBreak;
   FWBoolParameter            m_drawBreakPoints;

   // general colors
   mutable FWLongParameter   m_backgroundColor; // can be set via Ctr+b key binding
   FWLongParameter           m_gamma;

   // geom colors
   FWLongParameter     m_geomTransparency2D;
   FWLongParameter     m_geomTransparency3D;
   FWLongParameter*    m_geomColors[kFWGeomColorSize];

 
   std::auto_ptr<FWViewEnergyScale>  m_energyScale;

private:
   CmsShowCommon(const CmsShowCommon&); // stop default
   const CmsShowCommon& operator=(const CmsShowCommon&); // stop default
};


#endif
