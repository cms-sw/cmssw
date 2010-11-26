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
// $Id: CmsShowCommon.cc,v 1.10 2010/10/20 20:09:23 amraktad Exp $
//

// system include files
#include <boost/bind.hpp>

// user include files
#include "Fireworks/Core/interface/CmsShowCommon.h"
#include "Fireworks/Core/interface/FWEveView.h"


//
// constructors and destructor
//
CmsShowCommon::CmsShowCommon(FWColorManager* c):
   FWConfigurableParameterizable(2),
   m_colorManager(c),
   m_backgroundColor(this, "backgroundColIdx", 1l, 0l, 1000l),
   m_gamma(this, "brightness", 0l, -15l, 15l),
   m_geomTransparency2D(this, "Transparency 2D", long(c->geomTransparency(true)), 0l, 100l),
   m_geomTransparency3D(this, "Transparency 3D", long(c->geomTransparency(false)), 0l, 100l),
   m_energyScale(new FWViewEnergyScale("global", 2))
{
   char name[32];
   for (int i = 0; i < kFWGeomColorSize; ++i)
   {
      snprintf(name, 31, "GeometryColor %d ", i);
      m_geomColors[i] = new FWLongParameter(this, name   , long(c->geomColor(FWGeomColorIndex(i))), 1000l, 1100l);
   }   
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
   m_energyScale->addTo(oTo);
}

void
CmsShowCommon::setFrom(const FWConfiguration& iFrom)
{  
   for(const_iterator it =begin(), itEnd = end();
       it != itEnd;
       ++it) {
      (*it)->setFrom(iFrom);      
   }  
 
   if (iFrom.version() > 1)
      m_energyScale->setFrom(iFrom);

   // background
   m_colorManager->setBackgroundAndBrightness( FWColorManager::BackgroundColorIndex(m_backgroundColor.value()), m_gamma.value());

   // geom colors
   m_colorManager->setGeomTransparency( m_geomTransparency2D.value(), true);
   m_colorManager->setGeomTransparency( m_geomTransparency3D.value(), false);

   for (int i = 0; i < kFWGeomColorSize; ++i)
      m_colorManager->setGeomColor(FWGeomColorIndex(i), m_geomColors[i]->value());
}
