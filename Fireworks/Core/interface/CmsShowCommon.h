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
// $Id: CmsShowCommon.h,v 1.7 2010/09/27 10:46:00 amraktad Exp $
//

#include <sigc++/signal.h>
#include <sigc++/sigc++.h>

#include "Rtypes.h"

#include "Fireworks/Core/interface/FWConfigurableParameterizable.h"
#include "Fireworks/Core/interface/FWBoolParameter.h"
#include "Fireworks/Core/interface/FWLongParameter.h"
#include "Fireworks/Core/interface/FWEnumParameter.h"
#include "Fireworks/Core/interface/FWColorManager.h"

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

   long   getEnergyScaleMode()      const { return m_energyScaleMode.value(); }
   double getEnergyToHeightFixed()  const { return m_energyFixedValToHeight.value(); }
   double getEnergyMaxTowerHeight() const { return m_energyMaxTowerHeight.value(); }
   bool   getEnergyPlotEt()         const { return m_energyPlotEt.value(); }

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   virtual void setFrom(const FWConfiguration&);

   int gamma() { return m_gamma.value(); }
   void setGamma(int);
   void switchBackground();

   void setGeomColor(FWGeomColorIndex, Color_t);
   void setGeomTransparency(int val, bool projected);

   sigc::signal<void> scaleChanged_;

protected:
   const FWColorManager*   colorManager() const { return m_colorManager;}
   virtual void updateScales();

   // ---------- member data --------------------------------

   FWColorManager*     m_colorManager;

   // general colors
   mutable FWLongParameter   m_backgroundColor; // can be set via Ctr+b key binding
   FWLongParameter           m_gamma;

   // geom colors
   FWLongParameter     m_geomTransparency2D;
   FWLongParameter     m_geomTransparency3D;
   FWLongParameter*    m_geomColors[kFWGeomColorSize];

   // scales 
   FWBoolParameter    m_energyPlotEt;
   FWEnumParameter    m_energyScaleMode;
   FWDoubleParameter  m_energyFixedValToHeight;
   FWDoubleParameter  m_energyMaxTowerHeight;
   // FWDoubleParameter  m_energyCombinedSwitch;
private:
   CmsShowCommon(const CmsShowCommon&); // stop default
   const CmsShowCommon& operator=(const CmsShowCommon&); // stop default
};


#endif
