#include <iostream>

#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"

#include "CondFormats/EcalObjects/interface/EcalLaserAlphas.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosRef.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

EcalLaserDbService::EcalLaserDbService () 
  : 
  mAlphas (0),
  mAPDPNRatiosRef (0),
  mAPDPNRatios (0)
 {}



const EcalLaserAlphas* EcalLaserDbService::getAlphas () const {
  return mAlphas;
}

const EcalLaserAPDPNRatiosRef* EcalLaserDbService::getAPDPNRatiosRef () const {
  return mAPDPNRatiosRef;
}

const EcalLaserAPDPNRatios* EcalLaserDbService::getAPDPNRatios () const {
  return mAPDPNRatios;
}


float EcalLaserDbService::getLaserCorrection (DetId const & xid, edm::Timestamp const & iTime) const {
  
  float correctionFactor = 1.0;
  
  const EcalLaserAlphas* myalpha = mAlphas;         
  const EcalLaserAPDPNRatios* myapdpn = mAPDPNRatios;    
  const EcalLaserAPDPNRatiosRef* myapdpnref = mAPDPNRatiosRef; 
  
  const EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap& laserRatiosMap =  myapdpn->getLaserMap();
  const EcalLaserAPDPNRatios::EcalLaserTimeStampMap& laserTimeMap =  myapdpn->getTimeMap();
  const EcalLaserAPDPNRatiosRef::EcalLaserAPDPNRatiosRefMap& laserRefMap =  myapdpnref->getMap();
  const EcalLaserAlphas::EcalLaserAlphaMap& laserAlphaMap =  myalpha->getMap();
  
  EcalLaserAPDPNRatios::EcalLaserAPDPNpair apdpnpair;
  EcalLaserAPDPNRatios::EcalLaserTimeStamp timestamp;
  EcalLaserAPDPNRatiosRef::EcalLaserAPDPNref apdpnref;
  EcalLaserAlphas::EcalLaserAlpha alpha;
    
  EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMapIterator laserRatiosIter; // Laser iterator
  EcalLaserAPDPNRatios::EcalLaserTimeStampMapIterator laserTimeIter;
  EcalLaserAPDPNRatiosRef::EcalLaserAPDPNRatiosRefMapIterator laserRefIter; 
  EcalLaserAlphas::EcalLaserAlphaMapIterator laserAlphaIter; 	  

  
  if (xid.det()==DetId::Ecal) {
    std::cout << "XID is in Ecal" << std::endl;
  } else {
    std::cout << "XID is NOT in Ecal" << std::endl;
  }

  if (xid.subdetId()==EcalBarrel) {
    std::cout << "XID is EcalBarrel" << std::endl;
  } else if (xid.subdetId()==EcalEndcap) {
    std::cout << "XID is EcalEndcap" << std::endl;
  } else {
    std::cout << "XID is NOT EcalBarrel or EcalEndCap" << std::endl;
  }

  //  std::cout << "CRYSTAL  EBDetId: " << xid << " Timestamp: " << iTime.value() << std::endl;
  
  // get alpha, apd/pn ref, apd/pn pairs and timestamps for interpolation
  
  laserRatiosIter = laserRatiosMap.find(xid.rawId());
  if( laserRatiosIter != laserRatiosMap.end() ) {
    apdpnpair = laserRatiosIter->second;
    std::cout << " APDPN pair " << apdpnpair.p1 << " , " << apdpnpair.p2 << std::endl; 
  } else {
    edm::LogError("EcalLaserDbService") << "error with laserRatiosMap!" << endl;
    return correctionFactor;
  }
  
  // need to fix this so we map xtal with light monitoring modules!!

  int junkLM = 0;
  // this is junk for now
  if (xid.subdetId()==EcalBarrel) {
    EBDetId junkid(xid.rawId());
    junkLM = junkid.ism();
  } else if (xid.subdetId()==EcalEndcap) {
    EEDetId junkid(xid.rawId());
    junkLM = junkid.isc();
  } 

  std::cout << "JUNKLM ====> " << junkLM << endl;

  laserTimeIter = laserTimeMap.find(junkLM); // need to change this!
  if( laserTimeIter != laserTimeMap.end() ) {
    timestamp = laserTimeIter->second;
    std::cout << " TIME pair " << timestamp.t1 << " , " << timestamp.t2 << std::endl; 
  } else {
    edm::LogError("EcalLaserDbService") << "error with laserTimeMap!" << endl;
    return correctionFactor;
  }
  
  laserRefIter = laserRefMap.find(xid.rawId());
  if( laserRefIter != laserRefMap.end() ) {
    apdpnref = laserRefIter->second;
    std::cout << " APDPN ref " << apdpnref << std::endl; 
  } else {
    edm::LogError("EcalLaserDbService") << "error with laserRefMap!" << endl;
    return correctionFactor;
  }
  
  laserAlphaIter = laserAlphaMap.find(xid.rawId());
  if( laserAlphaIter != laserAlphaMap.end() ) {
    alpha = laserAlphaIter->second;    
    std::cout << " ALPHA " << alpha << std::endl; 
  } else {
    edm::LogError("EcalLaserDbService") << "error with laserAlphaMap!" << endl;
    return correctionFactor;
  }
  
  // should implement some default in case of error...

  // should do some quality checks first
  // ...

  // we will need to treat time differently...
  // is time in DB same format as in MC?  probably not...
  
  // interpolation

  if (apdpnref!=0&&(timestamp.t2-timestamp.t1)!=0) {
    float interpolatedLaserResponse = apdpnpair.p1 + (iTime.value()-timestamp.t1)*(apdpnpair.p2-apdpnpair.p1)/apdpnref/(timestamp.t2-timestamp.t1);
    std::cout << " interpolatedLaserResponse = " << interpolatedLaserResponse << std::endl; 

    if (interpolatedLaserResponse<=0) {
      edm::LogError("EcalLaserDbService") << "interpolatedLaserResponse is <= zero!" << endl;
      return correctionFactor;
    } else {
      correctionFactor = 1/pow(interpolatedLaserResponse,alpha);
    }
    
  } else {
    edm::LogError("EcalLaserDbService") << "apdpnref or timestamp.t2-timestamp.t1 is zero!" << endl;
    return correctionFactor;
  }
  
  std::cout << " correctionFactor = " << correctionFactor << std::endl; 

  correctionFactor = 1.0;  // set to correction of 1.0 for testing
  return correctionFactor;

}


EVENTSETUP_DATA_REG(EcalLaserDbService);
