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
// $Id: CmsShowCommon.h,v 1.8 2010/11/21 19:36:19 amraktad Exp $
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

class CmsShowCommon : public FWConfigurableParameterizable
{
   friend class CmsShowCommonPopup;

public:
   CmsShowCommon(FWColorManager*);
   virtual ~CmsShowCommon();

   // ---------- const member functions ---------------------
   virtual void addTo(FWConfiguration&) const;

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   virtual void setFrom(const FWConfiguration&);

   int gamma() { return m_gamma.value(); }
   void setGamma(int);
   void switchBackground();

   void setGeomColor(FWGeomColorIndex, Color_t);
   void setGeomTransparency(int val, bool projected);

   FWViewEnergyScale* getEnergyScale() const { return m_energyScale.get(); }

protected:
   const FWColorManager*   colorManager() const { return m_colorManager;}

   // ---------- member data --------------------------------

   FWColorManager*     m_colorManager;

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
