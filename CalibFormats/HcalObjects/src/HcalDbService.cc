//
// F.Ratnikov (UMd), Aug. 9, 2005
//

#include "FWCore/Utilities/interface/typelookup.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/QieShape.h"

#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrationWidths.h"

#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"

#include <cmath>

HcalDbService::HcalDbService (const edm::ParameterSet& cfg): 
  mPedestals (0), mPedestalWidths (0),
  mGains (0), mGainWidths (0),  
  mQieShapeCache (0), mQIEData(0), 
  mElectronicsMap(0),
  mRespCorrs(0),
  mL1TriggerObjects(0),
  mTimeCorrs(0),
  mLUTCorrs(0),
  mPFCorrs(0),
  mLutMetadata(0),
  mUpdateCalibrations (true), mUpdateCalibWidths(true)
 {}


const HcalCalibrations& HcalDbService::getHcalCalibrations(const HcalGenericDetId& fId) const 
{ 
  if (mUpdateCalibrations) 
    {
      buildCalibrations();
      mUpdateCalibrations = false;
    }
  return mCalibSet.getCalibrations(fId); 
}

const HcalCalibrationWidths& HcalDbService::getHcalCalibrationWidths(const HcalGenericDetId& fId) const 
{ 
  if (mUpdateCalibWidths) 
    {
      buildCalibWidths();
      mUpdateCalibWidths = false;
    }
  return mCalibWidthSet.getCalibrationWidths(fId); 
}

void HcalDbService::buildCalibrations() const {
  // we use the set of ids for pedestals as the master list
  if ((!mPedestals) || (!mGains) || (!mQIEData) || (!mRespCorrs) || (!mTimeCorrs) || (!mLUTCorrs) ) return;

  std::vector<DetId> ids=mPedestals->getAllChannels();
  bool pedsInADC = mPedestals->isADC();
  // clear the calibrations set
  mCalibSet.clear();
  // loop!
  HcalCalibrations tool;

  //  std::cout << " length of id-vector: " << ids.size() << std::endl;
  for (std::vector<DetId>::const_iterator id=ids.begin(); id!=ids.end(); ++id) {
    // make
    bool ok=makeHcalCalibration(*id,&tool,pedsInADC);
    // store
    if (ok) mCalibSet.setCalibrations(*id,tool);
    //    std::cout << "Hcal calibrations built... detid no. " << HcalGenericDetId(*id) << std::endl;
  }
  mCalibSet.sort();
}

void HcalDbService::buildCalibWidths() const {
  // we use the set of ids for pedestal widths as the master list
  if ((!mPedestalWidths) || (!mGainWidths) || (!mQIEData) ) return;

  std::vector<DetId> ids=mPedestalWidths->getAllChannels();
  bool pedsInADC = mPedestalWidths->isADC();
  // clear the calibrations set
  mCalibWidthSet.clear();
  // loop!
  HcalCalibrationWidths tool;

  //  std::cout << " length of id-vector: " << ids.size() << std::endl;
  for (std::vector<DetId>::const_iterator id=ids.begin(); id!=ids.end(); ++id) {
    // make
    bool ok=makeHcalCalibrationWidth(*id,&tool,pedsInADC);
    // store
    if (ok) mCalibWidthSet.setCalibrationWidths(*id,tool);
    //    std::cout << "Hcal calibrations built... detid no. " << HcalGenericDetId(*id) << std::endl;
  }
  mCalibWidthSet.sort();
}

bool HcalDbService::makeHcalCalibration (const HcalGenericDetId& fId, HcalCalibrations* fObject, bool pedestalInADC) const {
  if (fObject) {
    const HcalPedestal* pedestal = getPedestal (fId);
    const HcalGain* gain = getGain (fId);
    const HcalRespCorr* respcorr = getHcalRespCorr (fId);
    const HcalTimeCorr* timecorr = getHcalTimeCorr (fId);
    const HcalLUTCorr* lutcorr = getHcalLUTCorr (fId);

    if (pedestalInADC) {
      const HcalQIECoder* coder=getHcalCoder(fId);
      const HcalQIEShape* shape=getHcalShape(coder);
      if (pedestal && gain && shape && coder && respcorr && timecorr && lutcorr) {
	float pedTrue[4];
	for (int i=0; i<4; i++) {
	  float x=pedestal->getValues()[i];
	  int x1=(int)std::floor(x);
	  int x2=(int)std::floor(x+1);
	  // y = (y2-y1)/(x2-x1) * (x - x1) + y1  [note: x2-x1=1]
	  float y2=coder->charge(*shape,x2,i);
	  float y1=coder->charge(*shape,x1,i);
	  pedTrue[i]=(y2-y1)*(x-x1)+y1;
	}
	*fObject = HcalCalibrations (gain->getValues (), pedTrue, respcorr->getValue(), timecorr->getValue(), lutcorr->getValue() );
	return true; 
      }
    } else {
      if (pedestal && gain && respcorr && timecorr && lutcorr) {
	*fObject = HcalCalibrations (gain->getValues (), pedestal->getValues (), respcorr->getValue(), timecorr->getValue(), lutcorr->getValue() );
	return true;
      }
    }
  }
  return false;
}

bool HcalDbService::makeHcalCalibrationWidth (const HcalGenericDetId& fId, 
					      HcalCalibrationWidths* fObject, bool pedestalInADC) const {
  if (fObject) {
    const HcalPedestalWidth* pedestalwidth = getPedestalWidth (fId);
    const HcalGainWidth* gainwidth = getGainWidth (fId);
    if (pedestalInADC) {
      const HcalQIECoder* coder=getHcalCoder(fId);
      const HcalQIEShape* shape=getHcalShape(coder);
      if (pedestalwidth && gainwidth && shape && coder) {
	float pedTrueWidth[4];
	for (int i=0; i<4; i++) {
	  float x=pedestalwidth->getWidth(i);
	  // assume QIE is linear in low range and use x1=0 and x2=1
	  // y = (y2-y1) * (x) [do not add any constant, only scale!]
	  float y2=coder->charge(*shape,1,i);
	  float y1=coder->charge(*shape,0,i);
	  pedTrueWidth[i]=(y2-y1)*x;
	}
	*fObject = HcalCalibrationWidths (gainwidth->getValues (), pedTrueWidth);
	return true; 
      } 
    } else {
      if (pedestalwidth && gainwidth) {
	float pedestalWidth [4];
	for (int i = 0; i < 4; i++) pedestalWidth [i] = pedestalwidth->getWidth (i);
	*fObject = HcalCalibrationWidths (gainwidth->getValues (), pedestalWidth);
	return true;
      }      
    }
  }
  return false;
}  


const HcalRespCorr* HcalDbService::getHcalRespCorr (const HcalGenericDetId& fId) const {
  if (mRespCorrs) {
    return mRespCorrs->getValues (fId);
  }
  return 0;
}

const HcalPedestal* HcalDbService::getPedestal (const HcalGenericDetId& fId) const {
  if (mPedestals) {
    return mPedestals->getValues (fId);
  }
  return 0;
}

  const HcalPedestalWidth* HcalDbService::getPedestalWidth (const HcalGenericDetId& fId) const {
  if (mPedestalWidths) {
    return mPedestalWidths->getValues (fId);
  }
  return 0;
}

const HcalGain* HcalDbService::getGain (const HcalGenericDetId& fId) const {
  if (mGains) {
    return mGains->getValues(fId);
  }
  return 0;
}

  const HcalGainWidth* HcalDbService::getGainWidth (const HcalGenericDetId& fId) const {
  if (mGainWidths) {
    return mGainWidths->getValues (fId);
  }
  return 0;
}

const HcalQIECoder* HcalDbService::getHcalCoder (const HcalGenericDetId& fId) const {
  if (mQIEData) {
    return mQIEData->getCoder (fId);
  }
  return 0;
}

const HcalQIEShape* HcalDbService::getHcalShape (const HcalGenericDetId& fId) const {
  if (mQIEData) {
    return &mQIEData->getShape (fId);
  }
  return 0;
}

const HcalQIEShape* HcalDbService::getHcalShape (const HcalQIECoder *coder) const {
  if (mQIEData) {
    return &mQIEData->getShape(coder);
  }
  return 0;
}


const HcalElectronicsMap* HcalDbService::getHcalMapping () const {
  return mElectronicsMap;
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
  return 0;
}

const HcalLUTCorr* HcalDbService::getHcalLUTCorr (const HcalGenericDetId& fId) const {
  if (mLUTCorrs) {
    return mLUTCorrs->getValues (fId);
  }
  return 0;
}

const HcalPFCorr* HcalDbService::getHcalPFCorr (const HcalGenericDetId& fId) const {
  if (mPFCorrs) {
    return mPFCorrs->getValues (fId);
  }
  return 0;
}

const HcalLutMetadata* HcalDbService::getHcalLutMetadata () const {
  return mLutMetadata;
}

TYPELOOKUP_DATA_REG(HcalDbService);
