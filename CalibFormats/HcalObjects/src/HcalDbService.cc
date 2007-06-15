//
// F.Ratnikov (UMd), Aug. 9, 2005
//
// $Id: HcalDbService.cc,v 1.10 2006/04/13 22:40:41 fedor Exp $

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


HcalDbService::HcalDbService () 
  : 
  mQieShapeCache (0),
  mPedestals (0),
  mPedestalWidths (0),
  mGains (0),
  mGainWidths (0)
 {}

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
    return mGains->getValues (fId);
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
