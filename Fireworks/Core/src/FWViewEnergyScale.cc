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
// $Id: FWViewEnergyScale.cc,v 1.10 2010/11/27 22:08:24 amraktad Exp $
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


FWViewEnergyScale::FWViewEnergyScale(std::string name, int version):
FWConfigurableParameterizable(version),
m_scaleMode(this, "ScaleMode", 1l, 1l, 2l),
m_fixedValToHeight(this, "EnergyToLength [GeV/m]", 50.0, 1.0, 100.0),
m_maxTowerHeight(this, "MaximumLength [m]", 3.0, 0.01, 30.0 ),
m_plotEt(this, "PlotEt", true),
m_name(name),
m_scaleFactor3D(1.f),
m_scaleFactorLego(0.05f)
{
   m_scaleMode.addEntry(kFixedScale,   "FixedScale");
   m_scaleMode.addEntry(kAutoScale,    "AutomaticScale");
   m_scaleMode.addEntry(kCombinedScale,"CombinedScale");
   
   m_scaleMode.changed_.connect(boost::bind(&FWViewEnergyScale::scaleParameterChanged,this));
   m_fixedValToHeight.changed_.connect(boost::bind(&FWViewEnergyScale::scaleParameterChanged,this));
   m_maxTowerHeight.changed_.connect(boost::bind(&FWViewEnergyScale::scaleParameterChanged,this));
   m_plotEt.changed_.connect(boost::bind(&FWViewEnergyScale::scaleParameterChanged,this));
}

FWViewEnergyScale::~FWViewEnergyScale()
{
}

//________________________________________________________

void
FWViewEnergyScale::scaleParameterChanged() const
{
   parameterChanged_.emit();
}

float
FWViewEnergyScale::calculateScaleFactor(float iMaxVal, bool isLego) const
{ 
   // check if in combined mode
   int mode = m_scaleMode.value();
   if (mode == kCombinedScale)
   {
      mode = (m_maxTowerHeight.value() > 100*iMaxVal/m_fixedValToHeight.value()) ? kFixedScale : kAutoScale;   
      // printf("COMBINED  \n");
   }
   // get converison
 
   if (mode == kFixedScale)
   {
      //  printf("fixed mode %f \n",m_fixedValToHeight.value());
      // apply default constructor height 
      float length = isLego ? TMath::Pi() : 100;
      return length / m_fixedValToHeight.value();
   }
   else
   {
      float length = isLego ? TMath::Pi() : (100*m_maxTowerHeight.value()) ;
      // printf("[%d] length %f max %f  \n", isLego, length, iMaxVal);
      return length / iMaxVal;
   }
}


void
FWViewEnergyScale::updateScaleFactors(float iMaxVal) 
{ 
   m_scaleFactor3D   = calculateScaleFactor(iMaxVal, false);
   m_scaleFactorLego = calculateScaleFactor(iMaxVal, true);
}

void
FWViewEnergyScale::setFrom(const FWConfiguration& iFrom)
{
   for(const_iterator it =begin(), itEnd = end();
       it != itEnd;
       ++it) {
      (*it)->setFrom(iFrom);
   }  
}

void
FWViewEnergyScale::SetFromCmsShowCommonConfig(long mode, float convert, float maxH, bool et)
{
   m_scaleMode.set(mode);
   m_fixedValToHeight.set(convert);
   m_maxTowerHeight.set(maxH);
   m_plotEt.set(et > 0);
}
