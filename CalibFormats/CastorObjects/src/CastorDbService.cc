//
// F.Ratnikov (UMd), Aug. 9, 2005
// Adapted for CASTOR by L. Mundim
//
// $Id: CastorDbService.cc,v 1.4 2010/02/20 20:54:59 wmtan Exp $

#include "FWCore/Utilities/interface/typelookup.h"

#include "CalibFormats/CastorObjects/interface/CastorDbService.h"
#include "CalibFormats/CastorObjects/interface/CastorCoderDb.h"
#include "CalibFormats/CastorObjects/interface/QieShape.h"

#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"

#include <cmath>

CastorDbService::CastorDbService (const edm::ParameterSet& cfg)
  : 
  mQieShapeCache (0),
  mPedestals (0),
  mPedestalWidths (0),
  mGains (0),
  mGainWidths (0),
  mQIEData(0),
  mElectronicsMap(0)
 {}

bool CastorDbService::makeCastorCalibration (const HcalGenericDetId& fId, CastorCalibrations* fObject, bool pedestalInADC) const {
  if (fObject) {
    const CastorPedestal* pedestal = getPedestal (fId);
    const CastorGain* gain = getGain (fId);

    if (pedestalInADC) {
      const CastorQIEShape* shape=getCastorShape();
      const CastorQIECoder* coder=getCastorCoder(fId);
      if (pedestal && gain && shape && coder ) {
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
	*fObject = CastorCalibrations (gain->getValues (), pedTrue );
	return true; 
      }
    } else {
      if (pedestal && gain ) {
	*fObject = CastorCalibrations (gain->getValues (), pedestal->getValues () );
	return true;
      }
    }
  }
  return false;
}

void CastorDbService::buildCalibrations() {
  // we use the set of ids for pedestals as the master list
  if ((!mPedestals) || (!mGains) || (!mQIEData) ) return;
  std::vector<DetId> ids=mPedestals->getAllChannels();
  bool pedsInADC = mPedestals->isADC();
  // clear the calibrations set
  mCalibSet.clear();
  // loop!
  CastorCalibrations tool;

  //  std::cout << " length of id-vector: " << ids.size() << std::endl;
  for (std::vector<DetId>::const_iterator id=ids.begin(); id!=ids.end(); ++id) {
    // make
    bool ok=makeCastorCalibration(*id,&tool,pedsInADC);
    // store
    if (ok) mCalibSet.setCalibrations(*id,tool);
    //    std::cout << "Castor calibrations built... detid no. " << HcalGenericDetId(*id) << std::endl;
  }
  mCalibSet.sort();
}

void CastorDbService::buildCalibWidths() {
  // we use the set of ids for pedestal widths as the master list
  if ((!mPedestalWidths) || (!mGainWidths) || (!mQIEData) ) return;

  std::vector<DetId> ids=mPedestalWidths->getAllChannels();
  bool pedsInADC = mPedestalWidths->isADC();
  // clear the calibrations set
  mCalibWidthSet.clear();
  // loop!
  CastorCalibrationWidths tool;

  //  std::cout << " length of id-vector: " << ids.size() << std::endl;
  for (std::vector<DetId>::const_iterator id=ids.begin(); id!=ids.end(); ++id) {
    // make
    bool ok=makeCastorCalibrationWidth(*id,&tool,pedsInADC);
    // store
    if (ok) mCalibWidthSet.setCalibrationWidths(*id,tool);
    //    std::cout << "Castor calibrations built... detid no. " << HcalGenericDetId(*id) << std::endl;
  }
  mCalibWidthSet.sort();
}

bool CastorDbService::makeCastorCalibrationWidth (const HcalGenericDetId& fId, 
					      CastorCalibrationWidths* fObject, bool pedestalInADC) const {
  if (fObject) {
    const CastorPedestalWidth* pedestalwidth = getPedestalWidth (fId);
    const CastorGainWidth* gainwidth = getGainWidth (fId);
    if (pedestalInADC) {
      const CastorQIEShape* shape=getCastorShape();
      const CastorQIECoder* coder=getCastorCoder(fId);
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
	*fObject = CastorCalibrationWidths (gainwidth->getValues (), pedTrueWidth);
	return true; 
      } 
    } else {
      if (pedestalwidth && gainwidth) {
	float pedestalWidth [4];
	for (int i = 0; i < 4; i++) pedestalWidth [i] = pedestalwidth->getWidth (i);
	*fObject = CastorCalibrationWidths (gainwidth->getValues (), pedestalWidth);
	return true;
      }      
    }
  }
  return false;
}  


const CastorPedestal* CastorDbService::getPedestal (const HcalGenericDetId& fId) const {
  if (mPedestals) {
    return mPedestals->getValues (fId);
  }
  return 0;
}

  const CastorPedestalWidth* CastorDbService::getPedestalWidth (const HcalGenericDetId& fId) const {
  if (mPedestalWidths) {
    return mPedestalWidths->getValues (fId);
  }
  return 0;
}

const CastorGain* CastorDbService::getGain (const HcalGenericDetId& fId) const {
  if (mGains) {
    return mGains->getValues(fId);
  }
  return 0;
}

  const CastorGainWidth* CastorDbService::getGainWidth (const HcalGenericDetId& fId) const {
  if (mGainWidths) {
    return mGainWidths->getValues (fId);
  }
  return 0;
}

const CastorQIECoder* CastorDbService::getCastorCoder (const HcalGenericDetId& fId) const {
  if (mQIEData) {
    return mQIEData->getCoder (fId);
  }
  return 0;
}

const CastorQIEShape* CastorDbService::getCastorShape () const {
  if (mQIEData) {
    return &mQIEData->getShape ();
  }
  return 0;
}

const CastorElectronicsMap* CastorDbService::getCastorMapping () const {
  return mElectronicsMap;
}

const CastorChannelStatus* CastorDbService::getCastorChannelStatus (const HcalGenericDetId& fId) const
{
  return mChannelQuality->getValues (fId);
}

TYPELOOKUP_DATA_REG(CastorDbService);
