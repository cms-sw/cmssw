
//
// F.Ratnikov (UMd), Aug. 9, 2005
//

#ifndef HcalDbService_h
#define HcalDbService_h

#include <memory>
#include <map>

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrationWidths.h"
#include "CalibFormats/HcalObjects/interface/HcalChannelCoder.h"
#include "CalibFormats/HcalObjects/interface/QieShape.h"
#include "CalibFormats/HcalObjects/interface/HcalCoder.h"

class HcalDbServiceBase;

class HcalDbService {
 public:
  HcalDbService (const HcalDbServiceBase* fService);
  std::auto_ptr<HcalCalibrations> getHcalCalibrations (const cms::HcalDetId& fId) const;
  std::auto_ptr<HcalCalibrationWidths> getHcalCalibrationWidths (const cms::HcalDetId& fId) const;
  std::auto_ptr<HcalCoder> getHcalCoder (const cms::HcalDetId& fId) const;
  const QieShape* getBasicShape () const;
  std::auto_ptr<HcalChannelCoder> getChannelCoder (const cms::HcalDetId& fId) const;

  const HcalDbServiceBase* service () const {return mService;}
  
 private:
  mutable QieShape* mQieShape;
  const HcalDbServiceBase* mService;
};

#endif
