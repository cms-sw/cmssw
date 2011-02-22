// -*- C++ -*-
//
// Package:     Core
// Class  :     CmsShowCommon
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Fri Sep 10 14:50:32 CEST 2010
// $Id: CmsShowCommon.cc,v 1.9 2010/09/27 10:46:05 amraktad Exp $
//

// system include files
#include <boost/bind.hpp>

// user include files
#include "Fireworks/Core/interface/CmsShowCommon.h"
#include "Fireworks/Core/interface/FWEveView.h"

#include "Fireworks/Core/interface/FWViewEnergyScale.h"

//
// constructors and destructor
//
CmsShowCommon::CmsShowCommon(FWColorManager* c):
   m_colorManager(c),
   m_backgroundColor(this, "backgroundColIdx", 1l, 0l, 1000l),
   m_gamma(this, "brightness", 0l, -15l, 15l),
   m_geomTransparency2D(this, "Transparency 2D", long(c->geomTransparency(true)), 0l, 100l),
   m_geomTransparency3D(this, "Transparency 3D", long(c->geomTransparency(false)), 0l, 100l),

   m_energyPlotEt(this, "PlotEt", true),
   m_energyScaleMode(this, "ScaleMode", 1l, 1l, 2l),
   m_energyFixedValToHeight(this, "ValueToHeight [GeV/m]", 50.0, 1.0, 100.0 ),
   m_energyMaxTowerHeight(this, "MaxTowerH [m]", 1.0, 0.01, 3.0)
   // m_energyCombinedSwitch(this, "SwitchToAutomaticModeAbove [m]", 100.0, 1.0, 300.0)
{
   char name[32];
   for (int i = 0; i < kFWGeomColorSize; ++i)
   {
      snprintf(name, 31, "GeometryColor %d ", i);
      m_geomColors[i] = new FWLongParameter(this, name   , long(c->geomColor(FWGeomColorIndex(i))), 1000l, 1100l);
   }
   m_energyScaleMode.addEntry(FWViewEnergyScale::kFixedScale,   "FixedScale");
   m_energyScaleMode.addEntry(FWViewEnergyScale::kAutoScale,    "AutoScale");
   m_energyScaleMode.addEntry(FWViewEnergyScale::kCombinedScale,"CombinedScale");
   
   m_energyPlotEt.changed_.connect(boost::bind(&CmsShowCommon::updateScales,this));
   m_energyScaleMode.changed_.connect(boost::bind(&CmsShowCommon::updateScales,this));
   m_energyFixedValToHeight.changed_.connect(boost::bind(&CmsShowCommon::updateScales,this));
   m_energyMaxTowerHeight.changed_.connect(boost::bind(&CmsShowCommon::updateScales,this));
   //   m_energyCombinedSwitch.changed_.connect(boost::bind(&CmsShowCommon::updateScales,this));
}

CmsShowCommon::~CmsShowCommon()
{
}

//
// member functions
//

void
CmsShowCommon::setGamma(int x)
{
   m_colorManager->setBrightness(x);
   m_gamma.set(x);
}

void
CmsShowCommon::switchBackground()
{
  m_colorManager->switchBackground();
  m_backgroundColor.set(m_colorManager->background());
}

void
CmsShowCommon::setGeomColor(FWGeomColorIndex cidx, Color_t iColor)
{
   m_geomColors[cidx]->set(iColor);
   m_colorManager->setGeomColor(cidx, iColor);
}

void
CmsShowCommon::setGeomTransparency(int iTransp, bool projected)
{
   if (projected)
      m_geomTransparency2D.set(iTransp);
   else
      m_geomTransparency3D.set(iTransp);

   m_colorManager->setGeomTransparency(iTransp, projected);
}

/* Tell FWEveViewMangar that scales have changed */
void
CmsShowCommon::updateScales()
{
   scaleChanged_.emit();
}

//______________________________________________________________________________

void
CmsShowCommon::addTo(FWConfiguration& oTo) const
{
   m_backgroundColor.set(int(m_colorManager->background()));

   FWConfigurableParameterizable::addTo(oTo);
}

void
CmsShowCommon::setFrom(const FWConfiguration& iFrom)
{  
   // background
   FWConfigurableParameterizable::setFrom(iFrom);
   m_colorManager->setBackgroundAndBrightness( FWColorManager::BackgroundColorIndex(m_backgroundColor.value()), m_gamma.value());

   // geom colors
   m_colorManager->setGeomTransparency( m_geomTransparency2D.value(), true);
   m_colorManager->setGeomTransparency( m_geomTransparency3D.value(), false);

   for (int i = 0; i < kFWGeomColorSize; ++i)
      m_colorManager->setGeomColor(FWGeomColorIndex(i), m_geomColors[i]->value());

}
