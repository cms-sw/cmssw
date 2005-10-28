//
// F.Ratnikov (UMd), Aug. 9, 2005
//
// $Id: HcalDbService.cc,v 1.2 2005/10/05 00:37:56 fedor Exp $

#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalDetIdDb.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/QieShape.h"

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

std::auto_ptr <HcalCalibrations> HcalDbService::getHcalCalibrations (const HcalDetId& fId) const {
  if (mPedestals && mGains) {
    const float* gains =  mPedestals->getValues (HcalDetIdDb::HcalDetIdDb (fId));
    const float* pedestals = mGains->getValues (HcalDetIdDb::HcalDetIdDb (fId));
    if (gains && pedestals) {
      return std::auto_ptr <HcalCalibrations> (new HcalCalibrations (gains, pedestals));
    }
  }
  return std::auto_ptr <HcalCalibrations> (0);
}  

std::auto_ptr <HcalCalibrationWidths> HcalDbService::getHcalCalibrationWidths (const HcalDetId& fId) const {
  if (mPedestalWidths && mGainWidths) {
    const float* gainWidths =  mPedestalWidths->getValues (HcalDetIdDb::HcalDetIdDb (fId));
    const float* pedestalWidths = mGainWidths->getValues (HcalDetIdDb::HcalDetIdDb (fId));
    if (gainWidths && pedestalWidths) {
      return std::auto_ptr <HcalCalibrationWidths> (new HcalCalibrationWidths (gainWidths, pedestalWidths));
    }
  }
  return std::auto_ptr <HcalCalibrationWidths> (0);
}  

std::auto_ptr <HcalCoder> HcalDbService::getHcalCoder (const HcalDetId& fId) const {
  const QieShape* shape = getBasicShape ();
  std::auto_ptr<HcalChannelCoder> coder = getChannelCoder (fId);
  if (shape && coder.get ()) {
    return std::auto_ptr <HcalCoder> (new HcalCoderDb (*coder, *shape));
  }
  return std::auto_ptr <HcalCoder> (0);
}

const QieShape* HcalDbService::getBasicShape () const {
  if (!mQieShapeCache) { // get basic shape
    if (mQIEShape) {
      double bins [32];
      double binSizes [32];
      for (int i = 0; i < 32; i++) {
	bins [i] = mQIEShape->lowEdge (i);
	binSizes [i] = mQIEShape->lowEdge (i+1) - mQIEShape->lowEdge (i);
      }
      mQieShapeCache = new QieShape (bins, binSizes);
    }
  }
  return mQieShapeCache;
}

std::auto_ptr <HcalChannelCoder> HcalDbService::getChannelCoder (const HcalDetId& fId) const {
  if (mQIEData) {
    const float* offsets = mQIEData->getOffsets (HcalDetIdDb::HcalDetIdDb (fId));
    const float* slopes = mQIEData->getSlopes (HcalDetIdDb::HcalDetIdDb (fId));
    if (offsets && slopes) {
      return  std::auto_ptr <HcalChannelCoder> (new  HcalChannelCoder(offsets, slopes));
    }
  }
  return std::auto_ptr <HcalChannelCoder> ();
}


EVENTSETUP_DATA_REG(HcalDbService);
