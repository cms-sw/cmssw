
//
// F.Ratnikov (UMd), Aug. 9, 2005
// Adapted for CASTOR by L. Mundim
//

#ifndef CastorDbService_h
#define CastorDbService_h

#include <memory>
#include <map>

#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "CalibFormats/CastorObjects/interface/CastorChannelCoder.h"
#include "CalibFormats/CastorObjects/interface/QieShape.h"
#include "CalibFormats/CastorObjects/interface/CastorCoder.h"
#include "CalibFormats/CastorObjects/interface/CastorCalibrationsSet.h"
#include "CalibFormats/CastorObjects/interface/CastorCalibrationWidthsSet.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "CondFormats/CastorObjects/interface/AllObjects.h"

class CastorCalibrations;
class CastorCalibrationWidths;

class CastorDbService {
 public:
  CastorDbService (const edm::ParameterSet&);

  const CastorCalibrations& getCastorCalibrations(const HcalGenericDetId& fId) const 
  { return mCalibSet.getCalibrations(fId); }
  const CastorCalibrationWidths& getCastorCalibrationWidths(const HcalGenericDetId& fId) const 
  { return mCalibWidthSet.getCalibrationWidths(fId); }

  const CastorPedestal* getPedestal (const HcalGenericDetId& fId) const;
  const CastorPedestalWidth* getPedestalWidth (const HcalGenericDetId& fId) const;
  const CastorGain* getGain (const HcalGenericDetId& fId) const;
  const CastorGainWidth* getGainWidth (const HcalGenericDetId& fId) const;
  const CastorQIECoder* getCastorCoder (const HcalGenericDetId& fId) const;
  const CastorQIEShape* getCastorShape () const;
  const CastorElectronicsMap* getCastorMapping () const;
  const CastorChannelStatus* getCastorChannelStatus (const HcalGenericDetId& fId) const;

  void setData (const CastorPedestals* fItem) {mPedestals = fItem; buildCalibrations(); }
  void setData (const CastorPedestalWidths* fItem) {mPedestalWidths = fItem; buildCalibWidths(); }
  void setData (const CastorGains* fItem) {mGains = fItem; buildCalibrations(); }
  void setData (const CastorGainWidths* fItem) {mGainWidths = fItem; }
  void setData (const CastorQIEData* fItem) {mQIEData = fItem; }
  void setData (const CastorChannelQuality* fItem) {mChannelQuality = fItem;}
  void setData (const CastorElectronicsMap* fItem) {mElectronicsMap = fItem;}

 private:
  bool makeCastorCalibration (const HcalGenericDetId& fId, CastorCalibrations* fObject, 
			    bool pedestalInADC) const;
  void buildCalibrations();
  bool makeCastorCalibrationWidth (const HcalGenericDetId& fId, CastorCalibrationWidths* fObject, 
				 bool pedestalInADC) const;
  void buildCalibWidths();
  const CastorPedestals* mPedestals;
  const CastorPedestalWidths* mPedestalWidths;
  const CastorGains* mGains;
  const CastorGainWidths* mGainWidths;
  const CastorQIEData* mQIEData;
  const CastorChannelQuality* mChannelQuality;
  const CastorElectronicsMap* mElectronicsMap;
  CastorCalibrationsSet mCalibSet;
  CastorCalibrationWidthsSet mCalibWidthSet;
};

#endif
