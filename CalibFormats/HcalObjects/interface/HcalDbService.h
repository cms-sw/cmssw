#ifndef CalibFormats_HcalObjects_HcalDbService_h
#define CalibFormats_HcalObjects_HcalDbService_h

//
// F.Ratnikov (UMd), Aug. 9, 2005
//

#include <memory>
#include <map>
#include <atomic>

#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "CalibFormats/HcalObjects/interface/HcalCoder.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrationsSet.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrationWidthsSet.h"

#include "CondFormats/HcalObjects/interface/AllObjects.h"

class HcalCalibrations;
class HcalCalibrationWidths;
class HcalTopology;

class HcalDbService {
public:
  HcalDbService();
  ~HcalDbService();

  const HcalTopology* getTopologyUsed() const;

  const HcalCalibrations& getHcalCalibrations(const HcalGenericDetId& fId) const;
  const HcalCalibrationWidths& getHcalCalibrationWidths(const HcalGenericDetId& fId) const;
  const HcalCalibrationsSet* getHcalCalibrationsSet() const;
  const HcalCalibrationWidthsSet* getHcalCalibrationWidthsSet() const;

  const HcalPedestal* getPedestal(const HcalGenericDetId& fId) const;
  const HcalPedestalWidth* getPedestalWidth(const HcalGenericDetId& fId) const;
  const HcalPedestal* getEffectivePedestal(const HcalGenericDetId& fId) const;
  const HcalPedestalWidth* getEffectivePedestalWidth(const HcalGenericDetId& fId) const;
  const HcalGain* getGain(const HcalGenericDetId& fId) const;
  const HcalGainWidth* getGainWidth(const HcalGenericDetId& fId) const;
  const HcalQIECoder* getHcalCoder(const HcalGenericDetId& fId) const;
  const HcalQIEShape* getHcalShape(const HcalGenericDetId& fId) const;
  const HcalQIEShape* getHcalShape(const HcalQIECoder* coder) const;
  const HcalElectronicsMap* getHcalMapping() const;
  const HcalFrontEndMap* getHcalFrontEndMapping() const;
  const HcalRespCorr* getHcalRespCorr(const HcalGenericDetId& fId) const;
  const HcalTimeCorr* getHcalTimeCorr(const HcalGenericDetId& fId) const;
  const HcalL1TriggerObject* getHcalL1TriggerObject(const HcalGenericDetId& fId) const;
  const HcalChannelStatus* getHcalChannelStatus(const HcalGenericDetId& fId) const;
  const HcalZSThreshold* getHcalZSThreshold(const HcalGenericDetId& fId) const;
  const HcalLUTCorr* getHcalLUTCorr(const HcalGenericDetId& fId) const;
  const HcalPFCorr* getHcalPFCorr(const HcalGenericDetId& fId) const;
  const HcalLutMetadata* getHcalLutMetadata() const;
  const HcalQIEType* getHcalQIEType(const HcalGenericDetId& fId) const;
  const HcalSiPMParameter* getHcalSiPMParameter(const HcalGenericDetId& fId) const;
  const HcalSiPMCharacteristics* getHcalSiPMCharacteristics() const;
  const HcalTPChannelParameter* getHcalTPChannelParameter(const HcalGenericDetId& fId, bool throwOnFail = true) const;
  const HcalTPParameters* getHcalTPParameters() const;
  const HcalMCParam* getHcalMCParam(const HcalGenericDetId& fId) const;
  const HcalRecoParam* getHcalRecoParam(const HcalGenericDetId& fId) const;

  void setData(const HcalPedestals* fItem, bool eff = false) {
    if (eff)
      mEffectivePedestals = fItem;
    else
      mPedestals = fItem;
    mCalibSet = nullptr;
  }
  void setData(const HcalPedestalWidths* fItem, bool eff = false) {
    if (eff)
      mEffectivePedestalWidths = fItem;
    else
      mPedestalWidths = fItem;
    mCalibWidthSet = nullptr;
  }
  void setData(const HcalGains* fItem) {
    mGains = fItem;
    mCalibSet = nullptr;
  }
  void setData(const HcalGainWidths* fItem) {
    mGainWidths = fItem;
    mCalibWidthSet = nullptr;
  }
  void setData(const HcalQIEData* fItem) {
    mQIEData = fItem;
    mCalibSet = nullptr;
    mCalibWidthSet = nullptr;
  }
  void setData(const HcalQIETypes* fItem) {
    mQIETypes = fItem;
    mCalibSet = nullptr;
  }
  void setData(const HcalChannelQuality* fItem) { mChannelQuality = fItem; }
  void setData(const HcalElectronicsMap* fItem) { mElectronicsMap = fItem; }
  void setData(const HcalFrontEndMap* fItem) { mFrontEndMap = fItem; }
  void setData(const HcalRespCorrs* fItem) {
    mRespCorrs = fItem;
    mCalibSet = nullptr;
  }
  void setData(const HcalTimeCorrs* fItem) {
    mTimeCorrs = fItem;
    mCalibSet = nullptr;
  }
  void setData(const HcalZSThresholds* fItem) { mZSThresholds = fItem; }
  void setData(const HcalL1TriggerObjects* fItem) { mL1TriggerObjects = fItem; }
  void setData(const HcalLUTCorrs* fItem) {
    mLUTCorrs = fItem;
    mCalibSet = nullptr;
  }
  void setData(const HcalPFCorrs* fItem) { mPFCorrs = fItem; }
  void setData(const HcalLutMetadata* fItem) { mLutMetadata = fItem; }
  void setData(const HcalSiPMParameters* fItem) {
    mSiPMParameters = fItem;
    mCalibSet = nullptr;
  }
  void setData(const HcalSiPMCharacteristics* fItem) { mSiPMCharacteristics = fItem; }
  void setData(const HcalTPChannelParameters* fItem) {
    mTPChannelParameters = fItem;
    mCalibSet = nullptr;
  }
  void setData(const HcalTPParameters* fItem) { mTPParameters = fItem; }
  void setData(const HcalMCParams* fItem) { mMCParams = fItem; }
  void setData(const HcalRecoParams* fItem) { mRecoParams = fItem; }

private:
  bool makeHcalCalibration(const HcalGenericDetId& fId,
                           HcalCalibrations* fObject,
                           bool pedestalInADC,
                           bool effPedestalInADC) const;
  void buildCalibrations() const;
  bool makeHcalCalibrationWidth(const HcalGenericDetId& fId,
                                HcalCalibrationWidths* fObject,
                                bool pedestalInADC,
                                bool effPedestalInADC) const;
  void buildCalibWidths() const;
  bool convertPedestals(const HcalGenericDetId& fId, const HcalPedestal* pedestal, float* pedTrue, bool inADC) const;
  bool convertPedestalWidths(const HcalGenericDetId& fId,
                             const HcalPedestalWidth* pedestalwidth,
                             const HcalPedestal* pedestal,
                             float* pedTrueWidth,
                             bool inADC) const;
  const HcalPedestals* mPedestals;
  const HcalPedestalWidths* mPedestalWidths;
  const HcalPedestals* mEffectivePedestals;
  const HcalPedestalWidths* mEffectivePedestalWidths;
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
  const HcalRecoParams* mRecoParams;
  //  bool mPedestalInADC;
  mutable std::atomic<HcalCalibrationsSet const*> mCalibSet;
  mutable std::atomic<HcalCalibrationWidthsSet const*> mCalibWidthSet;
};

#endif
