//
// F.Ratnikov (UMd), Aug. 9, 2005
//
// $Id: HcalDbService.cc,v 1.12 2007/03/31 18:27:02 michals Exp $

#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/QieShape.h"

#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrationWidths.h"

#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "CondFormats/HcalObjects/interface/HcalGains.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidths.h"
#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"
#include "CondFormats/HcalObjects/interface/HcalQIEData.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "RecoLocalCalo/CaloTowersCreator/interface/EScales.h"


HcalDbService::HcalDbService () 
  : 
  mQieShapeCache (0),
  mPedestals (0),
  mPedestalWidths (0),
  mGains (0),
  mGainWidths (0)
 {}

HcalDbService::HcalDbService (const edm::ParameterSet& fConfig) 
  : 
  mQieShapeCache (0),
  mPedestals (0),
  mPedestalWidths (0),
  mGains (0),
  mGainWidths (0)
 {
  m_hbEScale = fConfig.getUntrackedParameter <double> ("hbEScale",1.);
  m_hesEScale = fConfig.getUntrackedParameter <double> ("hesEScale",1.);
  m_hedEScale = fConfig.getUntrackedParameter <double> ("hedEScale",1.);
  m_hoEScale = fConfig.getUntrackedParameter <double> ("hoEScale",1.);
  m_hf1EScale = fConfig.getUntrackedParameter <double> ("hf1EScale",1.);
  m_hf2EScale = fConfig.getUntrackedParameter <double> ("hf2EScale",1.);
  EScales.HBPiOvere = m_hbEScale;
  EScales.HESPiOvere = m_hesEScale;
  EScales.HEDPiOvere = m_hedEScale;
  EScales.HOPiOvere = m_hoEScale;
  EScales.HF1PiOvere = m_hf1EScale;
  EScales.HF2PiOvere = m_hf2EScale;
}

bool HcalDbService::makeHcalCalibration (const HcalGenericDetId& fId, HcalCalibrations* fObject) const {
  if (fObject) {
    const HcalPedestal* pedestal = getPedestal (fId);
    const HcalGain* gain = getGain (fId);
    if (pedestal && gain) {
      *fObject = HcalCalibrations (gain->getValues (), pedestal->getValues ());
      return true;
    }
  }
  return false;
}

bool HcalDbService::makeHcalCalibrationWidth (const HcalGenericDetId& fId, HcalCalibrationWidths* fObject) const {
  if (fObject) {
    const HcalPedestalWidth* pedestal = getPedestalWidth (fId);
    const HcalGainWidth* gain = getGainWidth (fId);
    if (pedestal && gain) {
      float pedestalWidth [4];
      for (int i = 0; i < 4; i++) pedestalWidth [i] = pedestal->getWidth (i);
      *fObject = HcalCalibrationWidths (gain->getValues (), pedestalWidth);
      return true;
    }
  }
  return false;
}  

const HcalPedestal* HcalDbService::getPedestal (const HcalGenericDetId& fId) const {
  if (mPedestals) {
    return mPedestals->getValues (fId);
  }
  return 0;
}

  const HcalPedestalWidth* HcalDbService::getPedestalWidth (const HcalGenericDetId& fId) const {
  if (mPedestalWidths) {
    return mPedestalWidths->getValues (fId);
  }
  return 0;
}

const HcalGain* HcalDbService::getGain (const HcalGenericDetId& fId) const {
  if (mGains) {
// make room to include fixed pion energy scale (read from cfi)
    float escale=0.;
    HcalDetId id(fId);
    if (fId.subdet()==HcalBarrel) escale=m_hbEScale;
    if (fId.subdet()==HcalEndcap && id.ietaAbs()<21) escale=m_hesEScale;
    if (fId.subdet()==HcalEndcap && id.ietaAbs()>=21) escale=m_hedEScale;
    if (fId.subdet()==HcalOuter) escale=m_hoEScale;
    if (fId.subdet()==HcalForward && id.depth()==1) escale=m_hf1EScale;
    if (fId.subdet()==HcalForward && id.depth()==2) escale=m_hf2EScale;
    float v0=escale*mGains->getValue(fId,0);
    float v1=escale*mGains->getValue(fId,1);
    float v2=escale*mGains->getValue(fId,2);
    float v3=escale*mGains->getValue(fId,3);
    HcalGains* newGains = new HcalGains();
    bool ok = newGains -> addValue(fId,v0,v1,v2,v3);
    newGains->sort();
    if (ok) return newGains->getValues(fId);
    else return 0;
  }
  return 0;
}

  const HcalGainWidth* HcalDbService::getGainWidth (const HcalGenericDetId& fId) const {
  if (mGainWidths) {
    return mGainWidths->getValues (fId);
  }
  return 0;
}

const HcalQIECoder* HcalDbService::getHcalCoder (const HcalGenericDetId& fId) const {
  if (mQIEData) {
    return mQIEData->getCoder (fId);
  }
  return 0;
}

const HcalQIEShape* HcalDbService::getHcalShape () const {
  if (mQIEData) {
    return &mQIEData->getShape ();
  }
  return 0;
}
const HcalElectronicsMap* HcalDbService::getHcalMapping () const {
  return mElectronicsMap;
}

EVENTSETUP_DATA_REG(HcalDbService);
