//
// F.Ratnikov (UMd), Aug. 9, 2005
//
// $Id: HcalDbService.cc,v 1.8 2005/12/27 23:50:27 fedor Exp $

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


namespace {
  unsigned long dbId (const HcalDetId& fId) {return fId.rawId ();}
}

HcalDbService::HcalDbService () 
  : 
  mQieShapeCache (0),
  mPedestals (0),
  mPedestalWidths (0),
  mGains (0),
  mGainWidths (0)
 {}

bool HcalDbService::makeHcalCalibration (const HcalDetId& fId, HcalCalibrations* fObject) const {
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

bool HcalDbService::makeHcalCalibrationWidth (const HcalDetId& fId, HcalCalibrationWidths* fObject) const {
  if (fObject) {
    const HcalPedestalWidth* pedestal = getPedestalWidth (fId);
    const HcalGainWidth* gain = getGainWidth (fId);
    if (pedestal && gain) {
      float pedestalWidth [4];
      for (int i = 0; i < 4; i++) pedestalWidth [i] = pedestal->getWidth (i+1);
      *fObject = HcalCalibrationWidths (gain->getValues (), pedestalWidth);
      return true;
    }
  }
  return false;
}  

const HcalPedestal* HcalDbService::getPedestal (const HcalDetId& fId) const {
  if (mPedestals) {
    return mPedestals->getValues (fId);
  }
  return 0;
}

  const HcalPedestalWidth* HcalDbService::getPedestalWidth (const HcalDetId& fId) const {
  if (mPedestalWidths) {
    return mPedestalWidths->getValues (fId);
  }
  return 0;
}

const HcalGain* HcalDbService::getGain (const HcalDetId& fId) const {
  if (mGains) {
    return mGains->getValues (fId);
  }
  return 0;
}

  const HcalGainWidth* HcalDbService::getGainWidth (const HcalDetId& fId) const {
  if (mGainWidths) {
    return mGainWidths->getValues (fId);
  }
  return 0;
}

const HcalQIECoder* HcalDbService::getHcalCoder (const HcalDetId& fId) const {
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
