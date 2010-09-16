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
// $Id: CmsShowCommon.cc,v 1.4 2010/09/16 17:31:53 amraktad Exp $
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
   m_geomTransparency2D(this, "geomTransparency2D", long(c->geomTransparency(true)), 0l, 100l),
   m_geomTransparency3D(this, "geomTransparency3D", long(c->geomTransparency(true)), 0l, 100l)
{
  
   m_geomColors[kFWMuonBarrelLineColorIndex] = new FWLongParameter(this, "muonBarrelColor", long(c->geomColor(kFWMuonBarrelLineColorIndex)), 1000l, 1100l);
   m_geomColors[kFWMuonEndcapLineColorIndex] = new FWLongParameter(this, "muonEndcapColor", long(c->geomColor(kFWMuonEndcapLineColorIndex)), 1000l, 1100l);
   m_geomColors[kFWTrackerColorIndex]        = new FWLongParameter(this, "trackerColor"   , long(c->geomColor(kFWTrackerColorIndex       )), 1000l, 1100l);
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
   m_backgroundColor.set(int(m_colorManager->background()));

   FWConfigurableParameterizable::addTo(oTo);
   //  printf("add %d %d %d \n", m_geomColors[0]->value(),m_geomColors[1]->value(),m_geomColors[2]->value() );
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
