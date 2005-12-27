//
// F.Ratnikov (UMd), Aug. 9, 2005
//
// $Id: HcalDbService.cc,v 1.7 2005/12/15 23:38:00 fedor Exp $

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
  if (mPedestals && mGains && fObject) {
    const float* pedestals =  mPedestals->getValues (fId)->getValues ();
    const float* gains = mGains->getValues (fId)->getValues ();
    if (gains && pedestals) {
      *fObject = HcalCalibrations (gains, pedestals);
      return true;
    }
  }
  return false;
}
bool HcalDbService::makeHcalCalibrationWidth (const HcalDetId& fId, HcalCalibrationWidths* fObject) const {
  if (mPedestalWidths && mGainWidths && fObject) {
    const float* pedestals =  mPedestalWidths->getValues (fId)->getValues ();
    const float* gains = mGainWidths->getValues (fId)->getValues ();
    if (gains && pedestals) {
      *fObject = HcalCalibrationWidths (gains, pedestals);
      return true;
    }
  }
  return false;
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
