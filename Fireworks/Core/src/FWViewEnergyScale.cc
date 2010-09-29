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
// $Id: FWViewEnergyScale.cc,v 1.4 2010/09/27 10:46:11 amraktad Exp $
//

#include <stdexcept>
#include <iostream>
#include <boost/bind.hpp>

#include "Rtypes.h"
#include "Fireworks/Core/interface/FWEveView.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/CmsShowCommon.h"


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
m_maxVal(0.f),
m_valToHeight(1.f),
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
FWViewEnergyScale::setValToHeight(float x)
{
   m_valToHeight = x;
}

float
FWViewEnergyScale::getValToHeight() const
{
   return m_valToHeight;  
}

bool 
FWViewEnergyScale::setMaxVal(float s)
{
   if (s > m_maxVal )
   {
      m_maxVal = s;
      return true;
   }
   
   return false;
}

float  
FWViewEnergyScale::getMaxVal() const
{
   return m_maxVal;
}

void  
FWViewEnergyScale::reset()
{
   m_maxVal = 0.f;
   m_valToHeight = 1.f;
}


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

double
FWViewEnergyScale::getMaxTowerHeight() const
{
  const static int m_to_cm = 100; // parameters in [m], TEveCaloViz in [cm]

   if (getUseGlobalScales())
      return m_to_cm * m_view->context().commonPrefs()->getEnergyMaxTowerHeight();
   else
      return m_to_cm * m_maxTowerHeight.value();
}

bool
FWViewEnergyScale::getPlotEt() const
{
   if (getUseGlobalScales())
      return m_view->context().commonPrefs()->getEnergyPlotEt();
   else
      return m_plotEt.value();
}
  
