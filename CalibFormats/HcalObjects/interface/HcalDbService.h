
//
// F.Ratnikov (UMd), Aug. 9, 2005
//

#ifndef HcalDbService_h
#define HcalDbService_h

#include <memory>
#include <map>

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CalibFormats/HcalObjects/interface/HcalChannelCoder.h"
#include "CalibFormats/HcalObjects/interface/QieShape.h"
#include "CalibFormats/HcalObjects/interface/HcalCoder.h"


class HcalCalibrations;
class HcalCalibrationWidths;

class HcalPedestal;
class HcalPedestalWidth;
class HcalGain;
class HcalGainWidth;
class HcalPedestals;
class HcalPedestalWidths;
class HcalGains;
class HcalGainWidths;
class HcalQIECoder;
class HcalQIEShape;
class HcalQIEData;
class HcalChannelQuality;
class HcalElectronicsMap;

class HcalDbService {
 public:
  HcalDbService ();

  bool makeHcalCalibration (const HcalDetId& fId, HcalCalibrations* fObject) const;
  bool makeHcalCalibrationWidth (const HcalDetId& fId, HcalCalibrationWidths* fObject) const;
  const HcalPedestal* getPedestal (const HcalDetId& fId) const;
  const HcalPedestalWidth* getPedestalWidth (const HcalDetId& fId) const;
  const HcalGain* getGain (const HcalDetId& fId) const;
  const HcalGainWidth* getGainWidth (const HcalDetId& fId) const;
  const HcalQIECoder* getHcalCoder (const HcalDetId& fId) const;
  const HcalQIEShape* getHcalShape () const;
  const HcalElectronicsMap* getHcalMapping () const;
  
  void setData (const HcalPedestals* fItem) {mPedestals = fItem;}
  void setData (const HcalPedestalWidths* fItem) {mPedestalWidths = fItem;}
  void setData (const HcalGains* fItem) {mGains = fItem;}
  void setData (const HcalGainWidths* fItem) {mGainWidths = fItem;}
  void setData (const HcalQIEData* fItem) {mQIEData = fItem;}
  void setData (const HcalChannelQuality* fItem) {mChannelQuality = fItem;}
  void setData (const HcalElectronicsMap* fItem) {mElectronicsMap = fItem;}
 private:
  mutable QieShape* mQieShapeCache;
  const HcalPedestals* mPedestals;
  const HcalPedestalWidths* mPedestalWidths;
  const HcalGains* mGains;
  const HcalGainWidths* mGainWidths;
  const HcalQIEData* mQIEData;
  const HcalChannelQuality* mChannelQuality;
  const HcalElectronicsMap* mElectronicsMap;
};

#endif
