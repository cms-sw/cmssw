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
// $Id$
//

// system include files

// user include files
#include "Fireworks/Core/interface/CmsShowCommon.h"
#include "Fireworks/Core/interface/FWColorManager.h"

//
// constructors and destructor
//
CmsShowCommon::CmsShowCommon(FWColorManager* c):
   m_colorManager(c),
   m_blackBackground(this, "blackBackground", true),
   m_gamma(this, "brightness", 0l, -15l, 15l)
{
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
  m_blackBackground.set(m_colorManager->isColorSetDark());
}

//______________________________________________________________________________

void
CmsShowCommon::addTo(FWConfiguration& oTo) const
{
   FWConfigurableParameterizable::addTo(oTo);
}

void
CmsShowCommon::setFrom(const FWConfiguration& iFrom)
{  
   FWConfigurableParameterizable::setFrom(iFrom);
   m_colorManager->setBackgroundAndBrightness( m_blackBackground.value() ? FWColorManager::kBlackIndex : FWColorManager::kWhiteIndex,
                                               m_gamma.value());
}
