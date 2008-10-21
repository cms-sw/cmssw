#ifndef CastorDbService_h
#define CastorDbService_h

#include <memory>
#include <map>

#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "CalibFormats/CastorObjects/interface/CastorChannelCoder.h"
#include "CalibFormats/CastorObjects/interface/QieShape.h"
#include "CalibFormats/CastorObjects/interface/CastorCoder.h"
#include "CalibFormats/CastorObjects/interface/CastorCalibrationsSet.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "CondFormats/CastorObjects/interface/AllObjects.h"

class CastorCalibrations;
class CastorCalibrationWidths;

class CastorPedestal;
class CastorPedestalWidth;
class CastorGain;
class CastorGainWidth;
class CastorPedestals;
class CastorPedestalWidths;
class CastorGains;
class CastorGainWidths;
class CastorQIECoder;
class CastorQIEShape;
class CastorQIEData;
class CastorChannelQuality;
class CastorElectronicsMap;

class CastorDbService {
 public:
  CastorDbService ();
  CastorDbService (const edm::ParameterSet&);

 bool makeCastorCalibration (const HcalGenericDetId& fId, CastorCalibrations* fObject) const;
 bool makeCastorCalibrationWidth (const HcalGenericDetId& fId, CastorCalibrationWidths* fObject) const;

 //add get method for Castor calibrations; affects reconstructor!
const CastorCalibrations& getCastorCalibrations(const HcalGenericDetId& fId) const { return mCalibSet.getCalibrations(fId); }

  const CastorPedestal* getPedestal (const HcalGenericDetId& fId) const;
  const CastorPedestalWidth* getPedestalWidth (const HcalGenericDetId& fId) const;
  const CastorGain* getGain (const HcalGenericDetId& fId) const;
  const CastorGainWidth* getGainWidth (const HcalGenericDetId& fId) const;
  const CastorQIECoder* getCastorCoder (const HcalGenericDetId& fId) const;
  const CastorQIEShape* getCastorShape () const;
  const CastorElectronicsMap* getCastorMapping () const;
  
  //  void setData (const CastorPedestals* fItem) {mPedestals = fItem; buildCalibrations();}
  void setData (const CastorPedestals* fItem) {mPedestals = fItem;}
  void setData (const CastorPedestalWidths* fItem) {mPedestalWidths = fItem;}
  void setData (const CastorGains* fItem) {mGains = fItem;}
  void setData (const CastorGainWidths* fItem) {mGainWidths = fItem;}
  //  void setData (const CastorQIEData* fItem) {mQIEData = fItem; buildCalibrations();}
  void setData (const CastorQIEData* fItem) {mQIEData = fItem;}
  void setData (const CastorChannelQuality* fItem) {mChannelQuality = fItem;}
  void setData (const CastorElectronicsMap* fItem) {mElectronicsMap = fItem;}

 private:
  //add methods for calibration; affect reconstructor!
  //  bool makeCastorCalibration (const HcalGenericDetId& fId, CastorCalibrations* fObject) const;
  //  void buildCalibrations();

  mutable QieShape* mQieShapeCache;
  const CastorPedestals* mPedestals;
  const CastorPedestalWidths* mPedestalWidths;
  const CastorGains* mGains;
  const CastorGainWidths* mGainWidths;
  const CastorQIEData* mQIEData;
  const CastorChannelQuality* mChannelQuality;
  const CastorElectronicsMap* mElectronicsMap;
  CastorCalibrationsSet mCalibSet;
};

#endif
