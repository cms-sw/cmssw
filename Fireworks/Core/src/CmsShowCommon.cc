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
// $Id: CmsShowCommon.cc,v 1.3 2010/09/15 18:14:22 amraktad Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/CmsShowCommon.h"

//
// constructors and destructor
//
CmsShowCommon::CmsShowCommon(FWColorManager* c):
   m_colorManager(c),
   m_backgroundColor(this, "backgroundColIdx", 1l, 0l, 1000l),
   m_gamma(this, "brightness", 0l, -15l, 15l),
   m_geomTransparency2D(this, "geomTransparency2D", 50l, 0l, 100l),
   m_geomTransparency3D(this, "geomTransparency3D", 70l, 0l, 100l)
{
   m_geomColors[kFWMuonBarrelLineColorIndex] = new FWLongParameter(this, "muonBarrelColor", 1020l, 1000l, 1100l);
   m_geomColors[kFWMuonEndcapLineColorIndex] = new FWLongParameter(this, "muonEndcapColor", 1017l, 1000l, 1100l);
   m_geomColors[kFWTrackerColorIndex]        = new FWLongParameter(this, "trackerColor"   , 1009l, 1000l, 1100l);
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


//______________________________________________________________________________

void
CmsShowCommon::addTo(FWConfiguration& oTo) const
{
   // TODO connected to signals
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
