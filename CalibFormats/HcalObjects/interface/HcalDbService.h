
//
// F.Ratnikov (UMd), Aug. 9, 2005
//

#ifndef HcalDbService_h
#define HcalDbService_h

#include <memory>
#include <map>
#include <atomic>

#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "CalibFormats/HcalObjects/interface/HcalCoder.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrationsSet.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrationWidthsSet.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "CondFormats/HcalObjects/interface/AllObjects.h"

class HcalCalibrations;
class HcalCalibrationWidths;
class HcalTopology;

class HcalDbService {
 public:
  HcalDbService (const edm::ParameterSet&);
  ~HcalDbService();

  const HcalTopology* getTopologyUsed() const;
  
  const HcalCalibrations& getHcalCalibrations(const HcalGenericDetId& fId) const;
  const HcalCalibrationWidths& getHcalCalibrationWidths(const HcalGenericDetId& fId) const;
  const HcalCalibrationsSet* getHcalCalibrationsSet() const;
  const HcalCalibrationWidthsSet* getHcalCalibrationWidthsSet() const;

  const HcalPedestal* getPedestal (const HcalGenericDetId& fId) const;
  const HcalPedestalWidth* getPedestalWidth (const HcalGenericDetId& fId) const;
  const HcalGain* getGain (const HcalGenericDetId& fId) const;
  const HcalGainWidth* getGainWidth (const HcalGenericDetId& fId) const;
  const HcalQIECoder* getHcalCoder (const HcalGenericDetId& fId) const;
  const HcalQIEShape* getHcalShape (const HcalGenericDetId& fId) const;
  const HcalQIEShape* getHcalShape (const HcalQIECoder *coder) const;
  const HcalElectronicsMap* getHcalMapping () const;
  const HcalFrontEndMap* getHcalFrontEndMapping () const;
  const HcalRespCorr* getHcalRespCorr (const HcalGenericDetId& fId) const;
  const HcalTimeCorr* getHcalTimeCorr (const HcalGenericDetId& fId) const;
  const HcalL1TriggerObject* getHcalL1TriggerObject (const HcalGenericDetId& fId) const;
  const HcalChannelStatus* getHcalChannelStatus (const HcalGenericDetId& fId) const;
  const HcalZSThreshold* getHcalZSThreshold (const HcalGenericDetId& fId) const;
  const HcalLUTCorr* getHcalLUTCorr (const HcalGenericDetId& fId) const;
  const HcalPFCorr* getHcalPFCorr (const HcalGenericDetId& fId) const;
  const HcalLutMetadata* getHcalLutMetadata () const;
  const HcalQIEType* getHcalQIEType (const HcalGenericDetId& fId) const;
  const HcalSiPMParameter* getHcalSiPMParameter (const HcalGenericDetId& fId) const;
  const HcalSiPMCharacteristics* getHcalSiPMCharacteristics () const;
  const HcalTPChannelParameter* getHcalTPChannelParameter (const HcalGenericDetId& fId) const;
  const HcalTPParameters* getHcalTPParameters () const;
  const HcalMCParam* getHcalMCParam (const HcalGenericDetId& fId) const;

  void setData (const HcalPedestals* fItem) {mPedestals = fItem; mCalibSet = nullptr;}
  void setData (const HcalPedestalWidths* fItem) {mPedestalWidths = fItem; mCalibWidthSet = nullptr;}
  void setData (const HcalGains* fItem) {mGains = fItem; mCalibSet = nullptr; }
  void setData (const HcalGainWidths* fItem) {mGainWidths = fItem; mCalibWidthSet = nullptr; }
  void setData (const HcalQIEData* fItem) {mQIEData = fItem; mCalibSet=nullptr; mCalibWidthSet=nullptr;}
  void setData (const HcalQIETypes* fItem) {mQIETypes = fItem; mCalibSet = nullptr; }
  void setData (const HcalChannelQuality* fItem) {mChannelQuality = fItem;}
  void setData (const HcalElectronicsMap* fItem) {mElectronicsMap = fItem;}
  void setData (const HcalFrontEndMap* fItem) {mFrontEndMap = fItem;}
  void setData (const HcalRespCorrs* fItem) {mRespCorrs = fItem; mCalibSet = nullptr; }
  void setData (const HcalTimeCorrs* fItem) {mTimeCorrs = fItem; mCalibSet = nullptr; }
  void setData (const HcalZSThresholds* fItem) {mZSThresholds = fItem;}
  void setData (const HcalL1TriggerObjects* fItem) {mL1TriggerObjects = fItem;}
  void setData (const HcalLUTCorrs* fItem) {mLUTCorrs = fItem; mCalibSet = nullptr; }
  void setData (const HcalPFCorrs* fItem) {mPFCorrs = fItem; }
  void setData (const HcalLutMetadata* fItem) {mLutMetadata = fItem;}
  void setData (const HcalSiPMParameters* fItem) {mSiPMParameters = fItem; mCalibSet = nullptr;}
  void setData (const HcalSiPMCharacteristics* fItem) {mSiPMCharacteristics = fItem;}
  void setData (const HcalTPChannelParameters* fItem) {mTPChannelParameters = fItem; mCalibSet = nullptr;}
  void setData (const HcalTPParameters* fItem) {mTPParameters = fItem;}
  void setData (const HcalMCParams* fItem) {mMCParams = fItem;}

 private:
  bool makeHcalCalibration (const HcalGenericDetId& fId, HcalCalibrations* fObject, 
			    bool pedestalInADC) const;
  void buildCalibrations() const;
  bool makeHcalCalibrationWidth (const HcalGenericDetId& fId, HcalCalibrationWidths* fObject, 
				 bool pedestalInADC) const;
  void buildCalibWidths() const;
  const HcalPedestals* mPedestals;
  const HcalPedestalWidths* mPedestalWidths;
  const HcalGains* mGains;
  const HcalGainWidths* mGainWidths;
  const HcalQIEData* mQIEData;
  const HcalQIETypes* mQIETypes;
  const HcalChannelQuality* mChannelQuality;
  const HcalElectronicsMap* mElectronicsMap;
  const HcalFrontEndMap* mFrontEndMap;
  const HcalRespCorrs* mRespCorrs;
  const HcalZSThresholds* mZSThresholds;
  const HcalL1TriggerObjects* mL1TriggerObjects;
  const HcalTimeCorrs* mTimeCorrs;
  const HcalLUTCorrs* mLUTCorrs;
  const HcalPFCorrs* mPFCorrs;
  const HcalLutMetadata* mLutMetadata;
  const HcalSiPMParameters* mSiPMParameters;
  const HcalSiPMCharacteristics* mSiPMCharacteristics;
  const HcalTPChannelParameters* mTPChannelParameters;
  const HcalTPParameters* mTPParameters;
  const HcalMCParams* mMCParams;
  //  bool mPedestalInADC;
  mutable std::atomic<HcalCalibrationsSet const *> mCalibSet;
  mutable std::atomic<HcalCalibrationWidthsSet const *> mCalibWidthSet;
};

#endif
