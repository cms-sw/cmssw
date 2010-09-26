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
// $Id: FWViewEnergyScale.h,v 1.2 2010/06/22 09:44:33 amraktad Exp $
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
   friend class FWEveViewScaleEditor;
   
public:
   enum EScaleMode { kFixedScale, kAutoScale, kCombinedScale, kNone };
   
   FWViewEnergyScale(FWEveView* v);
   virtual ~FWViewEnergyScale();

   // -- const functions
   float  getValToHeight() const;
   float  getMaxVal() const;
   
   bool   getUseGlobalScales() const { return m_useGlobalScales.value(); } 
   long   getScaleMode() const { return m_scaleMode.value(); }
   double getMaxFixedVal() const { return m_fixedValToHeight.value()*m_maxTowerHeight.value(); }
   double getMaxTowerHeight() const { return m_maxTowerHeight.value(); }
   bool   getPlotEt() const { return m_plotEt.value(); }
   
   FWEveView* getView() const { return m_view; }
   
   // -- memeber functions   
   void  setValToHeight(float);
   bool  setMaxVal(float);
   void  reset();

   
protected:
   // protected for fries class (editor)
   FWBoolParameter    m_useGlobalScales;
   FWEnumParameter    m_scaleMode;
   FWDoubleParameter  m_fixedValToHeight;
   FWDoubleParameter  m_maxTowerHeight;
   FWBoolParameter    m_plotEt;
   
private:
   FWViewEnergyScale(const FWViewEnergyScale&); // stop default

   const FWViewEnergyScale& operator=(const FWViewEnergyScale&); // stop default

   // ---------- member data --------------------------------
   float       m_maxVal;
   float       m_valToHeight;
   FWEveView*  m_view;
};


#endif
