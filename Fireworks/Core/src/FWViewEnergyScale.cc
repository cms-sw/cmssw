// -*- C++ -*-
//
// Package:     Core
// Class  :     FWViewEnergyScale
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Fri Jun 18 20:37:44 CEST 2010
// $Id: FWViewEnergyScale.cc,v 1.5 2010/09/29 16:19:49 amraktad Exp $
//

#include <stdexcept>
#include <iostream>
#include <boost/bind.hpp>

#include "Rtypes.h"
#include "TMath.h"
#include "Fireworks/Core/interface/FWEveView.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/CmsShowCommon.h"

float FWViewEnergyScale::s_initMaxVal = 100;

//
// constants, enums and typedefs
//
//
// static data member definitions
//

//
// constructors and destructor
//
FWViewEnergyScale::FWViewEnergyScale(FWEveView* view):
FWConfigurableParameterizable(view->version()),
m_useGlobalScales(this, "UseGlobalScales", true),
m_scaleMode(this, "ScaleMode", 1l, 1l, 2l),
m_fixedValToHeight(this, "ValueToHeight [GeV/m]", 50.0, 1.0, 100.0),
m_maxTowerHeight(this, "MaxTowerH [m]", 1.0, 0.01, 3.0 ),
m_plotEt(this, "PlotEt", true),
m_maxVal(s_initMaxVal),
m_view(view)
{
   m_useGlobalScales.changed_.connect(boost::bind(&FWEveView::updateEnergyScales, m_view));
   m_scaleMode.addEntry(kFixedScale,   "FixedScale");
   m_scaleMode.addEntry(kAutoScale,    "AutoScale");
   m_scaleMode.addEntry(kCombinedScale,"CombinedScale");
   m_scaleMode.changed_.connect(boost::bind(&FWEveView::updateEnergyScales,m_view));
   
   m_fixedValToHeight.changed_.connect(boost::bind(&FWEveView::updateEnergyScales,m_view));
   m_maxTowerHeight.changed_.connect(boost::bind(&FWEveView::updateEnergyScales,m_view));
   m_plotEt.changed_.connect(boost::bind(&FWEveView::updateEnergyScales,m_view));
}

FWViewEnergyScale::~FWViewEnergyScale()
{
}

void  
FWViewEnergyScale::reset()
{
   m_maxVal = s_initMaxVal;
}

bool 
FWViewEnergyScale::setMaxVal(float s)
{
   m_maxVal = s;   
   return false;
}

float  
FWViewEnergyScale::getMaxVal() const
{
   return m_maxVal;
}

//________________________________________________________
long   
FWViewEnergyScale::getScaleMode() const
{  
   if (getUseGlobalScales())
      return m_view->context().commonPrefs()->getEnergyScaleMode();
   else
      return m_scaleMode.value(); 
}

double
FWViewEnergyScale::getMaxFixedVal() const
{ 
   if (getUseGlobalScales())
      return m_view->context().commonPrefs()->getEnergyMaxAbsVal();
   else
      return m_fixedValToHeight.value()*m_maxTowerHeight.value();
}

bool
FWViewEnergyScale::getPlotEt() const
{
   if (getUseGlobalScales())
      return m_view->context().commonPrefs()->getEnergyPlotEt();
   else
      return m_plotEt.value();
}

double
FWViewEnergyScale::getMaxTowerHeight() const
{
   // lego XYZ dimensions are[ etaRng x 2Pi() x Pi() ]
   if (FWViewType::isLego(m_view->typeId()))
      return TMath::Pi();
   
   // RPZ and 3D views, include m->cm conversion
   if (getUseGlobalScales())
      return 100 * m_view->context().commonPrefs()->getEnergyMaxTowerHeight();
   else
      return 100 * m_maxTowerHeight.value();
}

//________________________________________________________

float
FWViewEnergyScale::getValToHeight() const
{ 
   // check if in combined mode
   int mode = getScaleMode();
   if (mode == kCombinedScale)
      mode = (m_maxVal >  getMaxFixedVal()) ? kFixedScale : kAutoScale;
   
   // get converison
   if (mode == kFixedScale)
      return getMaxTowerHeight() / getMaxFixedVal();
   else
      return getMaxTowerHeight() / m_maxVal;
}
