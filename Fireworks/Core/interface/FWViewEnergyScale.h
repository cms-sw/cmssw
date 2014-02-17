#ifndef Fireworks_Core_FWViewEnergyScale_h
#define Fireworks_Core_FWViewEnergyScale_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWViewEnergyScale
// 
/**\class FWViewEnergyScale FWViewEnergyScale.h Fireworks/Core/interface/FWViewEnergyScale.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Fri Jun 18 20:37:55 CEST 2010
// $Id: FWViewEnergyScale.h,v 1.8 2010/11/27 22:08:23 amraktad Exp $
//

// system include files

// user include files

#include "Fireworks/Core/interface/FWDoubleParameter.h"
#include "Fireworks/Core/interface/FWBoolParameter.h"
#include "Fireworks/Core/interface/FWLongParameter.h"
#include "Fireworks/Core/interface/FWEnumParameter.h"
#include "Fireworks/Core/interface/FWConfigurableParameterizable.h"

// forward declarations
class FWEveView;

class FWViewEnergyScale : public FWConfigurableParameterizable
{
   friend class FWViewEnergyScaleEditor;
   
public:
   enum EScaleMode { kFixedScale, kAutoScale, kCombinedScale, kNone };
   FWViewEnergyScale(std::string name, int version);
   virtual ~FWViewEnergyScale();

   void updateScaleFactors(float iMaxVal);

   float getScaleFactor3D()   const { return m_scaleFactor3D;   }
   float getScaleFactorLego() const { return m_scaleFactorLego; }

   bool  getPlotEt() const { return m_plotEt.value(); }

   void scaleParameterChanged() const;

   sigc::signal<void> parameterChanged_;

   // added for debug
   const std::string& name() const { return m_name; } 

   virtual void setFrom(const FWConfiguration&);
   void SetFromCmsShowCommonConfig(long mode, float convert, float maxH, bool et);

protected:
   FWEnumParameter    m_scaleMode;
   FWDoubleParameter  m_fixedValToHeight;
   FWDoubleParameter  m_maxTowerHeight;
   FWBoolParameter    m_plotEt;
   
private:
   FWViewEnergyScale(const FWViewEnergyScale&); // stop default
   const FWViewEnergyScale& operator=(const FWViewEnergyScale&); // stop default

   float calculateScaleFactor(float iMaxVal, bool isLego) const;

   const std::string m_name;

   // cached
   float m_scaleFactor3D;
   float m_scaleFactorLego;
};

#endif
