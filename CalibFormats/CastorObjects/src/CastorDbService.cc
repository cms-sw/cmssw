// ...CastorDbService...
// first draft copy from CalibFormats/CastorDbService.cc
//

#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

#include "CalibFormats/CastorObjects/interface/CastorDbService.h"
#include "CalibFormats/CastorObjects/interface/CastorCoderDb.h"
#include "CalibFormats/CastorObjects/interface/QieShape.h"

#include "CalibFormats/CastorObjects/interface/CastorCalibrations.h"
#include "CalibFormats/CastorObjects/interface/CastorCalibrationWidths.h"

#include "CondFormats/CastorObjects/interface/CastorPedestals.h"
#include "CondFormats/CastorObjects/interface/CastorPedestalWidths.h"
#include "CondFormats/CastorObjects/interface/CastorGains.h"
#include "CondFormats/CastorObjects/interface/CastorGainWidths.h"
#include "CondFormats/CastorObjects/interface/CastorQIEShape.h"
#include "CondFormats/CastorObjects/interface/CastorQIEData.h"
#include "CondFormats/CastorObjects/interface/CastorChannelQuality.h"
#include "CondFormats/CastorObjects/interface/CastorElectronicsMap.h"
#include "RecoLocalCalo/CaloTowersCreator/interface/EScales.h"


CastorDbService::CastorDbService () 
  : 
  mQieShapeCache (0),
  mPedestals (0),
  mPedestalWidths (0),
  mGains (0),
  mGainWidths (0)
 {}

bool CastorDbService::makeCastorCalibration (const HcalGenericDetId& fId, CastorCalibrations* fObject) const {
  if (fObject) {
    const CastorPedestal* pedestal = getPedestal (fId);
    const CastorGain* gain = getGain (fId);
    if (pedestal && gain) {
      *fObject = CastorCalibrations (gain->getValues (), pedestal->getValues ());
      return true;
    }
  }
  return false;
}

bool CastorDbService::makeCastorCalibrationWidth (const HcalGenericDetId& fId, CastorCalibrationWidths* fObject) const {
  if (fObject) {
    const CastorPedestalWidth* pedestal = getPedestalWidth (fId);
    const CastorGainWidth* gain = getGainWidth (fId);
    if (pedestal && gain) {
      float pedestalWidth [4];
      for (int i = 0; i < 4; i++) pedestalWidth [i] = pedestal->getWidth (i);
      *fObject = CastorCalibrationWidths (gain->getValues (), pedestalWidth);
      return true;
    }
  }
  return false;
}  

const CastorPedestal* CastorDbService::getPedestal (const HcalGenericDetId& fId) const {
  if (mPedestals) {
    return mPedestals->getValues (fId);
  }
  return 0;
}

  const CastorPedestalWidth* CastorDbService::getPedestalWidth (const HcalGenericDetId& fId) const {
  if (mPedestalWidths) {
    return mPedestalWidths->getValues (fId);
  }
  return 0;
}

const CastorGain* CastorDbService::getGain (const HcalGenericDetId& fId) const {
  if (mGains) {
    return mGains->getValues(fId);
  }
  return 0;
}

  const CastorGainWidth* CastorDbService::getGainWidth (const HcalGenericDetId& fId) const {
  if (mGainWidths) {
    return mGainWidths->getValues (fId);
  }
  return 0;
}

const CastorQIECoder* CastorDbService::getCastorCoder (const HcalGenericDetId& fId) const {
  if (mQIEData) {
    return mQIEData->getCoder (fId);
  }
  return 0;
}

const CastorQIEShape* CastorDbService::getCastorShape () const {
  if (mQIEData) {
    return &mQIEData->getShape ();
  }
  return 0;
}
const CastorElectronicsMap* CastorDbService::getCastorMapping () const {
  return mElectronicsMap;
}

EVENTSETUP_DATA_REG(CastorDbService);
