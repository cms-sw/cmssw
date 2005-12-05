
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
#include "CondFormats/HcalMapping/interface/HcalMapping.h"

class HcalPedestals;
class HcalPedestalWidths;
class HcalGains;
class HcalGainWidths;
class HcalQIEShape;
class HcalQIEData;
class HcalChannelQuality;
class HcalElectronicsMap;

class HcalDbService {
 public:
  HcalDbService ();
  std::auto_ptr<HcalCalibrations> getHcalCalibrations (const HcalDetId& fId) const;
  std::auto_ptr<HcalCalibrationWidths> getHcalCalibrationWidths (const HcalDetId& fId) const;
  std::auto_ptr<HcalCoder> getHcalCoder (const HcalDetId& fId) const;
  std::auto_ptr<HcalMapping> getHcalMapping () const;

  const QieShape* getBasicShape () const;
  std::auto_ptr<HcalChannelCoder> getChannelCoder (const HcalDetId& fId) const;
  
  void setData (const HcalPedestals* fItem) {mPedestals = fItem;}
  void setData (const HcalPedestalWidths* fItem) {mPedestalWidths = fItem;}
  void setData (const HcalGains* fItem) {mGains = fItem;}
  void setData (const HcalGainWidths* fItem) {mGainWidths = fItem;}
  void setData (const HcalQIEShape* fItem) {mQIEShape = fItem;}
  void setData (const HcalQIEData* fItem) {mQIEData = fItem;}
  void setData (const HcalChannelQuality* fItem) {mChannelQuality = fItem;}
  void setData (const HcalElectronicsMap* fItem) {mElectronicsMap = fItem;}
 private:
  mutable QieShape* mQieShapeCache;
  const HcalPedestals* mPedestals;
  const HcalPedestalWidths* mPedestalWidths;
  const HcalGains* mGains;
  const HcalGainWidths* mGainWidths;
  const HcalQIEShape* mQIEShape;
  const HcalQIEData* mQIEData;
  const HcalChannelQuality* mChannelQuality;
  const HcalElectronicsMap* mElectronicsMap;
};

#endif
