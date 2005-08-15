
//
// F.Ratnikov (UMd), Aug. 9, 2005
//

#ifndef HcalDbServiceHandle_h
#define HcalDbServiceHandle_h

#include <memory>
#include <map>

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrationWidths.h"

class HcalDbService;

class HcalDbServiceHandle {
 public:
  HcalDbServiceHandle (const HcalDbService* fService);
  const HcalCalibrations* getHcalCalibrations (const cms::HcalDetId& fId) const;
  const HcalCalibrationWidths* getHcalCalibrationWidths (const cms::HcalDetId& fId) const;

  const HcalDbService* service () const {return mService;}
  
 private:
  mutable std::map<uint32_t, HcalCalibrations> mCalibrations;
  mutable std::map<uint32_t, HcalCalibrationWidths> mCalibrationWidths;

  const HcalDbService* mService;
};

#endif
