
//
// F.Ratnikov (UMd), Aug. 9, 2005
//

#ifndef HcalDbService_h
#define HcalDbService_h

#include <memory>
#include <map>

#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "CalibFormats/HcalObjects/interface/HcalChannelCoder.h"
#include "CalibFormats/HcalObjects/interface/QieShape.h"
#include "CalibFormats/HcalObjects/interface/HcalCoder.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrationsSet.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

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
  HcalDbService (const edm::ParameterSet&);

  bool makeHcalCalibrationWidth (const HcalGenericDetId& fId, HcalCalibrationWidths* fObject) const;
  const HcalCalibrations& getHcalCalibrations(const HcalGenericDetId& fId) const { return mCalibSet.getCalibrations(fId); }

  const HcalPedestal* getPedestal (const HcalGenericDetId& fId) const;
  const HcalPedestalWidth* getPedestalWidth (const HcalGenericDetId& fId) const;
  const HcalGain* getGain (const HcalGenericDetId& fId) const;
  const HcalGainWidth* getGainWidth (const HcalGenericDetId& fId) const;
  const HcalQIECoder* getHcalCoder (const HcalGenericDetId& fId) const;
  const HcalQIEShape* getHcalShape () const;
  const HcalElectronicsMap* getHcalMapping () const;
  
  void setData (const HcalPedestals* fItem) {mPedestals = fItem; buildCalibrations(); }
  void setData (const HcalPedestalWidths* fItem) {mPedestalWidths = fItem;}
  void setData (const HcalGains* fItem) {mGains = fItem;}
  void setData (const HcalGainWidths* fItem) {mGainWidths = fItem;}
  void setData (const HcalQIEData* fItem) {mQIEData = fItem; buildCalibrations(); }
  void setData (const HcalChannelQuality* fItem) {mChannelQuality = fItem;}
  void setData (const HcalElectronicsMap* fItem) {mElectronicsMap = fItem;}
 private:
  bool makeHcalCalibration (const HcalGenericDetId& fId, HcalCalibrations* fObject) const;
  void buildCalibrations();
  mutable QieShape* mQieShapeCache;
  const HcalPedestals* mPedestals;
  const HcalPedestalWidths* mPedestalWidths;
  const HcalGains* mGains;
  const HcalGainWidths* mGainWidths;
  const HcalQIEData* mQIEData;
  const HcalChannelQuality* mChannelQuality;
  const HcalElectronicsMap* mElectronicsMap;
  bool mPedestalInADC;
  HcalCalibrationsSet mCalibSet;
};

#endif
