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
// $Id: FWViewEnergyScale.h,v 1.5 2010/11/21 11:18:13 amraktad Exp $
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
   double getMaxTowerHeight() const;
   bool   getPlotEt() const;  
   bool   getUseGlobalScales() const { return m_useGlobalScales.value(); }

   FWEveView* getView() const { return m_view; }
   
   // -- memeber functions   
   bool  setMaxVal(float);
   void  reset();

   //protected:
   // protected for friend editor class
   long   getScaleMode() const;
   double getValToHeightFixed() const;

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
   FWEveView*  m_view;
   
  static float s_initMaxVal;
};


#endif
