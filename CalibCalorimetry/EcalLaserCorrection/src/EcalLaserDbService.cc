#include <iostream>

#include "FWCore/Utilities/interface/typelookup.h"

#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"

#include "CalibCalorimetry/EcalLaserAnalyzer/interface/MEEBGeom.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/MEEEGeom.h"
// #include "CalibCalorimetry/EcalLaserAnalyzer/interface/ME.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

EcalLaserDbService::EcalLaserDbService () 
  : 
  mAlphas_ (0),
  mAPDPNRatiosRef_ (0),
  mAPDPNRatios_ (0),
  mLinearCorrections_ (0)
 {}



const EcalLaserAlphas* EcalLaserDbService::getAlphas () const {
  return mAlphas_;
}

const EcalLaserAPDPNRatiosRef* EcalLaserDbService::getAPDPNRatiosRef () const {
  return mAPDPNRatiosRef_;
}

const EcalLaserAPDPNRatios* EcalLaserDbService::getAPDPNRatios () const {
  return mAPDPNRatios_;
}

const EcalLinearCorrections* EcalLaserDbService::getLinearCorrections () const {
  return mLinearCorrections_;
}



float EcalLaserDbService::getLaserCorrection (DetId const & xid, edm::Timestamp const & iTime) const {
  
  float correctionFactor = 1.0;

  const EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap& laserRatiosMap =  mAPDPNRatios_->getLaserMap();
  const EcalLaserAPDPNRatios::EcalLaserTimeStampMap& laserTimeMap =  mAPDPNRatios_->getTimeMap();
  const EcalLaserAPDPNRatiosRefMap& laserRefMap =  mAPDPNRatiosRef_->getMap();
  const EcalLaserAlphaMap& laserAlphaMap =  mAlphas_->getMap();
  const EcalLinearCorrections::EcalValueMap& linearValueMap =  mLinearCorrections_->getValueMap();
  const EcalLinearCorrections::EcalTimeMap& linearTimeMap =  mLinearCorrections_->getTimeMap();

  EcalLaserAPDPNRatios::EcalLaserAPDPNpair apdpnpair;
  EcalLaserAPDPNRatios::EcalLaserTimeStamp timestamp;
  EcalLaserAPDPNref apdpnref;
  EcalLaserAlpha alpha;
  EcalLinearCorrections::Values linValues;
  EcalLinearCorrections::Times linTimes;

  if (xid.det()==DetId::Ecal) {

  } else {

    edm::LogError("EcalLaserDbService") << " DetId is NOT in ECAL" << endl;
    return correctionFactor;
  } 


  int iLM;
  int xind;
  bool isBarrel=true;
  if (xid.subdetId()==EcalBarrel) {
    EBDetId ebid( xid.rawId() );
    xind = ebid.hashedIndex();
    iLM = MEEBGeom::lmr(ebid.ieta(), ebid.iphi());
  } else if (xid.subdetId()==EcalEndcap) {
    isBarrel=false;
    EEDetId eeid( xid.rawId() );  
    xind = eeid.hashedIndex();
    // SuperCrystal coordinates
    MEEEGeom::SuperCrysCoord iX = (eeid.ix()-1)/5 + 1;
    MEEEGeom::SuperCrysCoord iY = (eeid.iy()-1)/5 + 1;    
    iLM = MEEEGeom::lmr(iX, iY, eeid.zside());    
  } else {
    edm::LogError("EcalLaserDbService") << " DetId is NOT in ECAL Barrel or Endcap" << endl;
    return correctionFactor;
  }
  //  std::cout << " LM num ====> " << iLM << endl;

  // get alpha, apd/pn ref, apd/pn pairs and timestamps for interpolation


#ifdef VERIFY_LASER
  EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap::const_iterator itratio = laserRatiosMap.find(xid);
  if (itratio != laserRatiosMap.end()) {
    apdpnpair = (*itratio);
  } else {
    edm::LogError("EcalLaserDbService") << "error with laserRatiosMap!" << endl;     
    return correctionFactor;
  }

  if (iLM-1< (int)laserTimeMap.size()) {
    timestamp = laserTimeMap[iLM-1];  
  } else {
    edm::LogError("EcalLaserDbService") << "error with laserTimeMap!" << endl;     
    return correctionFactor;
  }

  EcalLinearCorrections::EcalValueMap::const_iterator itlin = linearValueMap.find(xid);
  if (itlin != linearValueMap.end()) {
    linValues = (*itlin);
  } else {
    edm::LogError("EcalLaserDbService") << "error with linearValueMap!" << endl;     
    return correctionFactor;
  }

  if (iLM-1< (int)linearTimeMap.size()) {
    linTimes = linearTimeMap[iLM-1];  
  } else {
    edm::LogError("EcalLaserDbService") << "error with laserTimeMap!" << endl;     
    return correctionFactor;
  }

  EcalLaserAPDPNRatiosRefMap::const_iterator itref = laserRefMap.find(xid);
  if ( itref != laserRefMap.end() ) {
    apdpnref = (*itref);
  } else { 
    edm::LogError("EcalLaserDbService") << "error with laserRefMap!" << endl;     
    return correctionFactor;
  }

  EcalLaserAlphaMap::const_iterator italpha = laserAlphaMap.find(xid);
  if ( italpha != laserAlphaMap.end() ) {
    alpha = (*italpha);
  } else {
    edm::LogError("EcalLaserDbService") << "error with laserAlphaMap!" << endl;     
    return correctionFactor;
  }

#else
    
  // waiting for templated lambdas
  auto getCond =[=](EcalFloatCondObjectContainer const & cond)->float {
    return isBarrel ? cond.barrel(xind) : cond.endcap(xind);
  };

  auto getPair =[=](EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap const & cond)->EcalLaserAPDPNRatios::EcalLaserAPDPNpair {
    return isBarrel ? cond.barrel(xind) : cond.endcap(xind);
  };

  auto getLinear =[=](EcalLinearCorrections::EcalValueMap const & cond)->EcalLinearCorrections::Values {
    return isBarrel ? cond.barrel(xind) : cond.endcap(xind);
  };


  apdpnpair = getPair(laserRatiosMap);
  linValues = getLinear(linearValueMap);
  apdpnref  = getCond(laserRefMap);
  alpha     = getCond(laserAlphaMap);

  if (iLM-1< (int)laserTimeMap.size()) {
    timestamp = laserTimeMap[iLM-1];
  } else {
    edm::LogError("EcalLaserDbService") << "error with laserTimeMap!" << endl;
    return correctionFactor;
  }

  if (iLM-1< (int)linearTimeMap.size()) {
    linTimes = linearTimeMap[iLM-1];
  } else {
    edm::LogError("EcalLaserDbService") << "error with laserTimeMap!" << endl;
    return correctionFactor;
  }


#endif


  
  // should implement some default in case of error...

  // should do some quality checks first
  // ...

  // we will need to treat time differently...
  // is time in DB same format as in MC?  probably not...
  
  // interpolation

  edm::TimeValue_t t = iTime.value();
  long long t_i = 0, t_f = 0;
  float p_i = 0, p_f = 0;
  long long lt_i = 0, lt_f = 0;
  float lp_i = 0, lp_f = 0;

  if ( t >= timestamp.t1.value() && t < timestamp.t2.value() ) {
          t_i = timestamp.t1.value();
          t_f = timestamp.t2.value();
          p_i = apdpnpair.p1;
          p_f = apdpnpair.p2;
  } else if ( t >= timestamp.t2.value() && t <= timestamp.t3.value() ) {
          t_i = timestamp.t2.value();
          t_f = timestamp.t3.value();
          p_i = apdpnpair.p2;
          p_f = apdpnpair.p3;
  } else if ( t < timestamp.t1.value() ) {
          t_i = timestamp.t1.value();
          t_f = timestamp.t2.value();
          p_i = apdpnpair.p1;
          p_f = apdpnpair.p2;

  } else if ( t > timestamp.t3.value() ) {
          t_i = timestamp.t2.value();
          t_f = timestamp.t3.value();
          p_i = apdpnpair.p2;
          p_f = apdpnpair.p3;

  }

  if ( t >= linTimes.t1.value() && t < linTimes.t2.value() ) {
          lt_i = linTimes.t1.value();
          lt_f = linTimes.t2.value();
          lp_i = linValues.p1;
          lp_f = linValues.p2;
  } else if ( t >= linTimes.t2.value() && t <= linTimes.t3.value() ) {
          lt_i = linTimes.t2.value();
          lt_f = linTimes.t3.value();
          lp_i = linValues.p2;
          lp_f = linValues.p3;
  } else if ( t < linTimes.t1.value() ) {
          lt_i = linTimes.t1.value();
          lt_f = linTimes.t2.value();
          lp_i = linValues.p1;
          lp_f = linValues.p2;

  } else if ( t > linTimes.t3.value() ) {
          lt_i = linTimes.t2.value();
          lt_f = linTimes.t3.value();
          lp_i = linValues.p2;
          lp_f = linValues.p3;

  }

  if ( apdpnref != 0 && (t_i - t_f) != 0 && (lt_i - lt_f) != 0) {
    long long tt = t; // never subtract two unsigned!
    float interpolatedLaserResponse = p_i/apdpnref + float(tt-t_i)*(p_f-p_i)/(apdpnref*float(t_f-t_i)); 
    float interpolatedLinearResponse = lp_i/apdpnref + float(tt-lt_i)*(lp_f-lp_i)/(apdpnref*float(lt_f-lt_i)); // FIXED BY FC
    
    if(interpolatedLinearResponse >2.f || interpolatedLinearResponse <0.1f) 
		interpolatedLinearResponse=1.f;
    if ( interpolatedLaserResponse <= 0. ) {

		// print message only if it is the first time we see < 0
        // on this detid
		if(channelsWithInvalidCorrection_.insert(xid.rawId()).second) {
			edm::LogError("EcalLaserDbService")
				<< "Interpolated Laser correction <0 for detid "<<xid.rawId(); 

		}
	
		return correctionFactor;	

    } else {

      float interpolatedTransparencyResponse = interpolatedLaserResponse / interpolatedLinearResponse;

      correctionFactor =  1.f/( std::pow(interpolatedTransparencyResponse,alpha) *interpolatedLinearResponse  );
      
    }
    
  } else {
    edm::LogError("EcalLaserDbService") 
            << "apdpnref (" << apdpnref << ") "
            << "or t_i-t_f (" << (t_i - t_f) << " is zero!";
    return correctionFactor;
  }
  
  return correctionFactor;
}




TYPELOOKUP_DATA_REG(EcalLaserDbService);
