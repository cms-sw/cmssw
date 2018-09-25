//
// F.Ratnikov (UMd), Aug. 9, 2005
//

#include "FWCore/Utilities/interface/typelookup.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrationWidths.h"

#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"

#include <cmath>

HcalDbService::HcalDbService() :
  mPedestals (nullptr), mPedestalWidths (nullptr),
  mEffectivePedestals (nullptr), mEffectivePedestalWidths (nullptr),
  mGains (nullptr), mGainWidths (nullptr),
  mQIEData(nullptr),
  mQIETypes(nullptr),
  mElectronicsMap(nullptr), mFrontEndMap(nullptr),
  mRespCorrs(nullptr),
  mL1TriggerObjects(nullptr),
  mTimeCorrs(nullptr),
  mLUTCorrs(nullptr),
  mPFCorrs(nullptr),
  mLutMetadata(nullptr),
  mSiPMParameters(nullptr), mSiPMCharacteristics(nullptr),
  mTPChannelParameters(nullptr), mTPParameters(nullptr),
  mMCParams(nullptr), mRecoParams(nullptr),
  mCalibSet(nullptr), mCalibWidthSet(nullptr)
 {}

HcalDbService::~HcalDbService() {
    delete mCalibSet.load();
    delete mCalibWidthSet.load();
}

const HcalTopology* HcalDbService::getTopologyUsed() const {
  if (mPedestals && mPedestals->topo()) return mPedestals->topo();
  if (mEffectivePedestals && mEffectivePedestals->topo()) return mEffectivePedestals->topo();
  if (mGains && mGains->topo())         return mGains->topo();
  if (mRespCorrs && mRespCorrs->topo()) return mRespCorrs->topo();
  if (mQIETypes && mQIETypes->topo())   return mQIETypes->topo();
  if (mL1TriggerObjects && mL1TriggerObjects->topo()) return mL1TriggerObjects->topo();
  if (mLutMetadata && mLutMetadata->topo()) return mLutMetadata->topo();
  return nullptr;
}


const HcalCalibrations& HcalDbService::getHcalCalibrations(const HcalGenericDetId& fId) const
{
  buildCalibrations();
  return (*mCalibSet.load(std::memory_order_acquire)).getCalibrations(fId);
}

const HcalCalibrationWidths& HcalDbService::getHcalCalibrationWidths(const HcalGenericDetId& fId) const
{
  buildCalibWidths();
  return (*mCalibWidthSet.load(std::memory_order_acquire)).getCalibrationWidths(fId);
}

const HcalCalibrationsSet* HcalDbService::getHcalCalibrationsSet() const
{
  buildCalibrations();
  return mCalibSet.load(std::memory_order_acquire);
}

const HcalCalibrationWidthsSet* HcalDbService::getHcalCalibrationWidthsSet() const
{
  buildCalibWidths();
  return mCalibWidthSet.load(std::memory_order_acquire);
}

void HcalDbService::buildCalibrations() const {
  // we use the set of ids for pedestals as the master list
  if ((!mPedestals) || (!mEffectivePedestals) || (!mGains) || (!mQIEData) || (!mQIETypes) || (!mRespCorrs) || (!mTimeCorrs) || (!mLUTCorrs) ) return;

  if (!mCalibSet.load(std::memory_order_acquire)) {

      auto ptr = new HcalCalibrationsSet();

      std::vector<DetId> ids=mPedestals->getAllChannels();
      bool pedsInADC = mPedestals->isADC();
      bool effPedsInADC = mEffectivePedestals->isADC();
      // loop!
      HcalCalibrations tool;

      //  std::cout << " length of id-vector: " << ids.size() << std::endl;
      for (std::vector<DetId>::const_iterator id=ids.begin(); id!=ids.end(); ++id) {
        // make
        bool ok=makeHcalCalibration(*id,&tool,pedsInADC,effPedsInADC);
        // store
        if (ok) ptr->setCalibrations(*id,tool);
        //    std::cout << "Hcal calibrations built... detid no. " << HcalGenericDetId(*id) << std::endl;
      }
      HcalCalibrationsSet const * cptr = ptr;
      HcalCalibrationsSet const * expect = nullptr;
      bool exchanged = mCalibSet.compare_exchange_strong(expect, cptr, std::memory_order_acq_rel);
      if(!exchanged) {
          delete ptr;
      }
  }
}

void HcalDbService::buildCalibWidths() const {
  // we use the set of ids for pedestal widths as the master list
  if ((!mPedestalWidths) || (!mEffectivePedestalWidths) || (!mGainWidths) || (!mQIEData) ) return;

  if (!mCalibWidthSet.load(std::memory_order_acquire)) {

      auto ptr = new HcalCalibrationWidthsSet();

      const std::vector<DetId>& ids=mPedestalWidths->getAllChannels();
      bool pedsInADC = mPedestalWidths->isADC();
      bool effPedsInADC = mEffectivePedestalWidths->isADC();
      // loop!
      HcalCalibrationWidths tool;

      //  std::cout << " length of id-vector: " << ids.size() << std::endl;
      for (std::vector<DetId>::const_iterator id=ids.begin(); id!=ids.end(); ++id) {
        // make
        bool ok=makeHcalCalibrationWidth(*id,&tool,pedsInADC,effPedsInADC);
        // store
        if (ok) ptr->setCalibrationWidths(*id,tool);
        //    std::cout << "Hcal calibrations built... detid no. " << HcalGenericDetId(*id) << std::endl;
      }
      HcalCalibrationWidthsSet const *  cptr =	ptr;
      HcalCalibrationWidthsSet const * expect = nullptr;
      bool exchanged = mCalibWidthSet.compare_exchange_strong(expect, cptr, std::memory_order_acq_rel);
      if(!exchanged) {
          delete ptr;
      }
  }
}

bool HcalDbService::convertPedestals(const HcalGenericDetId& fId, const HcalPedestal* pedestal, float* pedTrue, bool inADC) const {
  if(!pedestal) return false;
  const HcalQIECoder* coder = nullptr;
  const HcalQIEShape* shape = nullptr;
  if(inADC){
    coder = getHcalCoder(fId);
    shape = getHcalShape(coder);
    if(!coder || !shape) return false;
  }
  for (int i=0; i<4; i++) {
    float x=pedestal->getValues()[i];
    if(!inADC){
      pedTrue[i] = x;
      continue;
    }
    int x1=(int)std::floor(x);
    int x2=(int)std::floor(x+1);
    float y2=coder->charge(*shape,x2,i);
    float y1=coder->charge(*shape,x1,i);
    pedTrue[i]=(y2-y1)*(x-x1)+y1;
  }
  return true;
}

bool HcalDbService::makeHcalCalibration (const HcalGenericDetId& fId, HcalCalibrations* fObject, bool pedestalInADC, bool effPedestalInADC) const {
  if (fObject) {
    const HcalPedestal* pedestal = getPedestal (fId);
    const HcalPedestal* effpedestal = getEffectivePedestal (fId);
    const HcalGain* gain = getGain (fId);
    const HcalRespCorr* respcorr = getHcalRespCorr (fId);
    const HcalTimeCorr* timecorr = getHcalTimeCorr (fId);
    const HcalLUTCorr* lutcorr = getHcalLUTCorr (fId);

    float pedTrue[4];
    bool converted = convertPedestals(fId,pedestal,pedTrue,pedestalInADC);

    float effPedTrue[4];
    bool effconverted = convertPedestals(fId,effpedestal,effPedTrue,effPedestalInADC);

    if (pedestal && effpedestal && converted && effconverted && gain && respcorr && timecorr && lutcorr) {
      *fObject = HcalCalibrations (gain->getValues(), pedTrue, effPedTrue, respcorr->getValue(), timecorr->getValue(), lutcorr->getValue() );
      return true;
    }
  }
  return false;
}

bool HcalDbService::convertPedestalWidths(const HcalGenericDetId& fId, const HcalPedestalWidth* pedestalwidth, float* pedTrueWidth, bool inADC) const {
  if(!pedestalwidth) return false;
  const HcalQIECoder* coder = nullptr;
  const HcalQIEShape* shape = nullptr;
  if(inADC){
    coder = getHcalCoder(fId);
    shape = getHcalShape(coder);
    if(!coder || !shape) return false;
  }
  for (int i=0; i<4; i++) {
    float x=pedestalwidth->getWidth(i);
    if(!inADC){
      pedTrueWidth[i] = x;
      continue;
    }
    // assume QIE is linear in low range and use x1=0 and x2=1
    // y = (y2-y1) * (x) [do not add any constant, only scale!]
    float y2=coder->charge(*shape,1,i);
    float y1=coder->charge(*shape,0,i);
    pedTrueWidth[i]=(y2-y1)*x;
  }
  return true;
}


bool HcalDbService::makeHcalCalibrationWidth (const HcalGenericDetId& fId,
					      HcalCalibrationWidths* fObject, bool pedestalInADC, bool effPedestalInADC) const {
  if (fObject) {
    const HcalPedestalWidth* pedestalwidth = getPedestalWidth (fId);
    const HcalPedestalWidth* effpedestalwidth = getEffectivePedestalWidth (fId);
    const HcalGainWidth* gainwidth = getGainWidth (fId);

    float pedTrueWidth[4];
    bool converted = convertPedestalWidths(fId,pedestalwidth,pedTrueWidth,pedestalInADC);

    float effPedTrueWidth[4];
    bool effconverted = convertPedestalWidths(fId,effpedestalwidth,effPedTrueWidth,effPedestalInADC);
    if (pedestalwidth && effpedestalwidth&& gainwidth && converted && effconverted) {
      *fObject = HcalCalibrationWidths (gainwidth->getValues (), pedTrueWidth, effPedTrueWidth);
      return true;
    }
  }
  return false;
}

const HcalQIEType* HcalDbService::getHcalQIEType (const HcalGenericDetId& fId) const {
  if (mQIETypes) {
    return mQIETypes->getValues (fId);
  }
  return nullptr;
}

const HcalRespCorr* HcalDbService::getHcalRespCorr (const HcalGenericDetId& fId) const {
  if (mRespCorrs) {
    return mRespCorrs->getValues (fId);
  }
  return nullptr;
}

const HcalPedestal* HcalDbService::getPedestal (const HcalGenericDetId& fId) const {
  if (mPedestals) {
    return mPedestals->getValues (fId);
  }
  return nullptr;
}

const HcalPedestalWidth* HcalDbService::getPedestalWidth (const HcalGenericDetId& fId) const {
  if (mPedestalWidths) {
    return mPedestalWidths->getValues (fId);
  }
  return nullptr;
}

const HcalPedestal* HcalDbService::getEffectivePedestal (const HcalGenericDetId& fId) const {
  if (mEffectivePedestals) {
    return mEffectivePedestals->getValues (fId);
  }
  return nullptr;
}

const HcalPedestalWidth* HcalDbService::getEffectivePedestalWidth (const HcalGenericDetId& fId) const {
  if (mEffectivePedestalWidths) {
    return mEffectivePedestalWidths->getValues (fId);
  }
  return nullptr;
}

const HcalGain* HcalDbService::getGain (const HcalGenericDetId& fId) const {
  if (mGains) {
    return mGains->getValues(fId);
  }
  return nullptr;
}

  const HcalGainWidth* HcalDbService::getGainWidth (const HcalGenericDetId& fId) const {
  if (mGainWidths) {
    return mGainWidths->getValues (fId);
  }
  return nullptr;
}

const HcalQIECoder* HcalDbService::getHcalCoder (const HcalGenericDetId& fId) const {
  if (mQIEData) {
    return mQIEData->getCoder (fId);
  }
  return nullptr;
}

const HcalQIEShape* HcalDbService::getHcalShape (const HcalGenericDetId& fId) const {
  if (mQIEData && mQIETypes) {
    //currently 3 types of QIEs exist: QIE8, QIE10, QIE11
    int qieType = mQIETypes->getValues(fId)->getValue();
    //QIE10 and QIE11 have same shape (ADC ladder)
    if(qieType>0) qieType = 1;
    return &mQIEData->getShape(qieType);
  }
  return nullptr;
}

const HcalQIEShape* HcalDbService::getHcalShape (const HcalQIECoder *coder) const {
  HcalGenericDetId fId(coder->rawId());
  return getHcalShape(fId);
}

const HcalElectronicsMap* HcalDbService::getHcalMapping () const {
  return mElectronicsMap;
}

const HcalFrontEndMap* HcalDbService::getHcalFrontEndMapping () const {
  return mFrontEndMap;
}

const HcalL1TriggerObject* HcalDbService::getHcalL1TriggerObject (const HcalGenericDetId& fId) const
{
  return mL1TriggerObjects->getValues (fId);
}

const HcalChannelStatus* HcalDbService::getHcalChannelStatus (const HcalGenericDetId& fId) const
{
  return mChannelQuality->getValues (fId);
}

const HcalZSThreshold* HcalDbService::getHcalZSThreshold (const HcalGenericDetId& fId) const
{
  return mZSThresholds->getValues (fId);
}

const HcalTimeCorr* HcalDbService::getHcalTimeCorr (const HcalGenericDetId& fId) const {
  if (mTimeCorrs) {
    return mTimeCorrs->getValues (fId);
  }
  return nullptr;
}

const HcalLUTCorr* HcalDbService::getHcalLUTCorr (const HcalGenericDetId& fId) const {
  if (mLUTCorrs) {
    return mLUTCorrs->getValues (fId);
  }
  return nullptr;
}

const HcalPFCorr* HcalDbService::getHcalPFCorr (const HcalGenericDetId& fId) const {
  if (mPFCorrs) {
    return mPFCorrs->getValues (fId);
  }
  return nullptr;
}

const HcalLutMetadata* HcalDbService::getHcalLutMetadata () const {
  return mLutMetadata;
}

const HcalSiPMParameter* HcalDbService::getHcalSiPMParameter (const HcalGenericDetId& fId) const {
  if (mSiPMParameters) {
    return mSiPMParameters->getValues (fId);
  }
  return nullptr;
}

const HcalSiPMCharacteristics* HcalDbService::getHcalSiPMCharacteristics () const {
  return mSiPMCharacteristics;
}

const HcalTPChannelParameter* HcalDbService::getHcalTPChannelParameter (const HcalGenericDetId& fId) const {
  if (mTPChannelParameters) {
    return mTPChannelParameters->getValues (fId);
  }
  return nullptr;
}

const HcalMCParam* HcalDbService::getHcalMCParam (const HcalGenericDetId& fId) const {
  if (mMCParams) {
    return mMCParams->getValues (fId);
  }
  return nullptr;
}

const HcalRecoParam* HcalDbService::getHcalRecoParam (const HcalGenericDetId& fId) const {
  if (mRecoParams) {
    return mRecoParams->getValues (fId);
  }
  return nullptr;
}

const HcalTPParameters* HcalDbService::getHcalTPParameters () const {
  return mTPParameters;
}

TYPELOOKUP_DATA_REG(HcalDbService);
