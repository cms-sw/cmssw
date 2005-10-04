//
// F.Ratnikov (UMd), Aug. 9, 2005
//
// $Id: HcalDbService.cc,v 1.2 2005/08/18 23:41:41 fedor Exp $

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/QieShape.h"
#include "CalibFormats/HcalObjects/interface/HcalDbServiceBase.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

HcalDbService::HcalDbService (const HcalDbServiceBase* fService) 
  : 
  mQieShape (0),
  mService (fService)
 {}

std::auto_ptr <HcalCalibrations> HcalDbService::getHcalCalibrations (const cms::HcalDetId& fId) const {
  const float* gains = mService->gains (fId);
  const float* pedestals = mService->pedestals (fId);
  if (!gains || !pedestals) {
    return std::auto_ptr <HcalCalibrations> (0);
  }
  return std::auto_ptr <HcalCalibrations> (new HcalCalibrations (gains, pedestals));
}  

std::auto_ptr <HcalCalibrationWidths> HcalDbService::getHcalCalibrationWidths (const cms::HcalDetId& fId) const {
  const float* gains = mService->gainErrors (fId);
  const float* pedestals = mService->pedestalErrors (fId);
  if (!gains || !pedestals) {
    return std::auto_ptr <HcalCalibrationWidths> (0);
  }
  return std::auto_ptr <HcalCalibrationWidths> (new HcalCalibrationWidths (gains, pedestals));
}  

std::auto_ptr <HcalCoder> HcalDbService::getHcalCoder (const cms::HcalDetId& fId) const {
  return std::auto_ptr <HcalCoder> (new HcalCoderDb (*getChannelCoder (fId), *getBasicShape ()));
}

const QieShape* HcalDbService::getBasicShape () const {
  if (!mQieShape) { // get basic shape
    double bins [32];
    double binSizes [32];
    for (int i = 0; i < 32; i++) {
      bins [i] = mService->adcShape (i);
      binSizes [i] = mService->adcShapeBin (i);
    }
    mQieShape = new QieShape (bins, binSizes);
  }
  return mQieShape;
}

std::auto_ptr <HcalChannelCoder> HcalDbService::getChannelCoder (const cms::HcalDetId& fId) const {
  const float* offset = mService->offsets (fId);
  const float* slope = mService->slopes (fId);
  if (!offset || !slope) {
    return std::auto_ptr <HcalChannelCoder> ();
  }
  return  std::auto_ptr <HcalChannelCoder> (new  HcalChannelCoder(offset, slope));
}


EVENTSETUP_DATA_REG(HcalDbService);
