//
// F.Ratnikov (UMd), Aug. 9, 2005
//
// $Id: HcalDbServiceHandle.cc,v 1.1 2005/08/17 18:51:44 fedor Exp $

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/QieShape.h"
#include "CalibFormats/HcalObjects/interface/HcalDbServiceHandle.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

HcalDbServiceHandle::HcalDbServiceHandle (const HcalDbService* fService) 
  : 
  mQieShape (0),
  mService (fService)
 {}

const HcalCalibrations* HcalDbServiceHandle::getHcalCalibrations (const cms::HcalDetId& fId) const {
  uint32_t id = fId.rawId ();
  std::map<uint32_t, HcalCalibrations>::iterator cell = mCalibrations.find (id);
  if (cell == mCalibrations.end ()) { // try to retrieve from DB
    double gain[4];
    double pedestal [4];
    for (int i = 0; i < 4; i++) {
      gain [i] = mService->gain (fId, i);
    }
    for (int i = 0; i < 4; i++) {
      pedestal [i] = mService->pedestal (fId, i);
    }
    cell = mCalibrations.insert(std::map<uint32_t, HcalCalibrations>::value_type (id, HcalCalibrations (gain, pedestal))).first;
  }
  return &(cell->second);
}

const HcalCalibrationWidths* HcalDbServiceHandle::getHcalCalibrationWidths (const cms::HcalDetId& fId) const {
  uint32_t id = fId.rawId ();
  std::map<uint32_t, HcalCalibrationWidths>::iterator cell = mCalibrationWidths.find (id);
  if (cell == mCalibrationWidths.end ()) { // try to retrieve from DB
    double gain[4];
    double pedestal [4];
    for (int i = 0; i < 4; i++) {
      gain [i] = mService->gain (fId, i);
    }
    for (int i = 0; i < 4; i++) {
      pedestal [i] = mService->pedestal (fId, i);
    }
    cell = mCalibrationWidths.insert(std::map<uint32_t, HcalCalibrationWidths>::value_type (id, HcalCalibrationWidths (gain, pedestal))).first;
  }
  return &(cell->second);
}

std::auto_ptr <HcalCoder> HcalDbServiceHandle::getHcalCoder (const cms::HcalDetId& fId) const {
  return std::auto_ptr <HcalCoder> (new HcalCoderDb (*getChannelCoder (fId), *getBasicShape ()));
}

const QieShape* HcalDbServiceHandle::getBasicShape () const {
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

const HcalChannelCoder* HcalDbServiceHandle::getChannelCoder (const cms::HcalDetId& fId) const {
  uint32_t id = fId.rawId ();
  std::map<uint32_t, HcalChannelCoder>::iterator cell = mChannelCoders.find (id);
  if (cell == mChannelCoders.end ()) { // try to retrieve from DB
    double offset [4][4];
    double slope [4][4];
    for (int range = 0; range < 4; range++) {
      for (int capId = 0; capId < 4; capId++) {
	offset [capId][range] = mService->offset (fId, capId, range);
	slope [capId][range] = mService->slope (fId, capId, range);
      }
    }
    cell = mChannelCoders.insert(std::map<uint32_t, HcalChannelCoder>::value_type (id, HcalChannelCoder (offset, slope))).first;
  }
  
  return &(cell->second);
}


EVENTSETUP_DATA_REG(HcalDbServiceHandle);
