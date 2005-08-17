//
// F.Ratnikov (UMd), Aug. 9, 2005
//
// $Id: HcalDbServiceHandle.cc,v 1.2 2005/07/14 21:57:26 wmtan Exp $

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbServiceHandle.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

HcalDbServiceHandle::HcalDbServiceHandle (const HcalDbService* fService) 
  : mService (fService) {}

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


EVENTSETUP_DATA_REG(HcalDbServiceHandle);
