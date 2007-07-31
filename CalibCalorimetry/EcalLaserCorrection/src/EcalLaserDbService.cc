#include <iostream>

#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"

#include "CondFormats/EcalObjects/interface/EcalLaserAlphas.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosRef.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

EcalLaserDbService::EcalLaserDbService () 
  : 
  mAlphas_ (0),
  mAPDPNRatiosRef_ (0),
  mAPDPNRatios_ (0)
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


float EcalLaserDbService::getLaserCorrection (DetId const & xid, edm::Timestamp const & iTime) const {
  
  float correctionFactor = 1.0;

  const EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap& laserRatiosMap =  mAPDPNRatios_->getLaserMap();
  const EcalLaserAPDPNRatios::EcalLaserTimeStampMap& laserTimeMap =  mAPDPNRatios_->getTimeMap();
  const EcalLaserAPDPNRatiosRef::EcalLaserAPDPNRatiosRefMap& laserRefMap =  mAPDPNRatiosRef_->getMap();
  const EcalLaserAlphas::EcalLaserAlphaMap& laserAlphaMap =  mAlphas_->getMap();

  EcalLaserAPDPNRatios::EcalLaserAPDPNpair apdpnpair;
  EcalLaserAPDPNRatios::EcalLaserTimeStamp timestamp;
  EcalLaserAPDPNRatiosRef::EcalLaserAPDPNref apdpnref;
  EcalLaserAlphas::EcalLaserAlpha alpha;
    
  EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMapIterator laserRatiosIter; // Laser iterator
  EcalLaserAPDPNRatios::EcalLaserTimeStampMapIterator laserTimeIter;
  EcalLaserAPDPNRatiosRef::EcalLaserAPDPNRatiosRefMapIterator laserRefIter; 
  EcalLaserAlphas::EcalLaserAlphaMapIterator laserAlphaIter; 	  

  if (xid.det()==DetId::Ecal) {
    std::cout << " XID is in Ecal : ";
  } else {
    std::cout << " XID is NOT in Ecal : ";
    edm::LogError("EcalLaserDbService") << " DetId is NOT in ECAL" << endl;
    return correctionFactor;
  } 

  if (xid.subdetId()==EcalBarrel) {
    std::cout << "EcalBarrel" << std::endl;
  } else if (xid.subdetId()==EcalEndcap) {
    std::cout << "EcalEndcap" << std::endl;
  } else {
    std::cout << "NOT EcalBarrel or EcalEndCap" << std::endl;
    edm::LogError("EcalLaserDbService") << " DetId is NOT in ECAL Barrel or Endcap" << endl;
    return correctionFactor;
  }

  // need to fix this so we map xtal with light monitoring modules!!
  int iLM = getLMNumber(xid);
  //  std::cout << " LM num ====> " << iLM << endl;

  // get alpha, apd/pn ref, apd/pn pairs and timestamps for interpolation
  
  laserRatiosIter = laserRatiosMap.find(xid.rawId());
  if( laserRatiosIter != laserRatiosMap.end() ) {
    apdpnpair = laserRatiosIter->second;
  } else {
    edm::LogError("EcalLaserDbService") << "error with laserRatiosMap!" << endl;
    return correctionFactor;
  }  

  laserTimeIter = laserTimeMap.find(iLM); // need to change this!
  if( laserTimeIter != laserTimeMap.end() ) {
    timestamp = laserTimeIter->second;
  } else {
    //    std::cout << " TIME pair error " << iLM << std::endl;
    edm::LogError("EcalLaserDbService") << "error with laserTimeMap! " << iLM << endl;
    return correctionFactor;
  }
  
  laserRefIter = laserRefMap.find(xid.rawId());
  if( laserRefIter != laserRefMap.end() ) {
    apdpnref = laserRefIter->second;
  } else {
    edm::LogError("EcalLaserDbService") << "error with laserRefMap!" << endl;
    return correctionFactor;
  }
  
  laserAlphaIter = laserAlphaMap.find(xid.rawId());
  if( laserAlphaIter != laserAlphaMap.end() ) {
    alpha = laserAlphaIter->second;    
  } else {
    edm::LogError("EcalLaserDbService") << "error with laserAlphaMap!" << endl;
    return correctionFactor;
  }

  std::cout << " APDPN pair " << apdpnpair.p1 << " , " << apdpnpair.p2 << std::endl; 
  std::cout << " TIME pair " << timestamp.t1 << " , " << timestamp.t2 << " iLM " << iLM << std::endl; 
  std::cout << " APDPN ref " << apdpnref << std::endl; 
  std::cout << " ALPHA " << alpha << std::endl; 
  
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

//
// function to map EB or EE xtal to light monitoring module readout number
// (should eventually port this code as a map in a more appropriate package)
//
int EcalLaserDbService::getLMNumber(DetId const & xid) const {

  int iLM = 0;

  if (xid.subdetId()==EcalBarrel) {
    
    EBDetId tempid(xid.rawId());
    
    int iSM  = tempid.ism();
    int iETA = tempid.ietaSM();
    int iPHI = tempid.iphiSM();
    
    const int nSM = 36;     
    int numLM[nSM] = {37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35};
    
    if (iSM<=nSM) {
      iLM = numLM[iSM-1];
    } 
    // now assign light module within SM
    if (iPHI>10&&iETA>5) { iLM++; }   

    std::cout << " SM , LM ---> " << iSM << " " << iLM << std::endl;
    
  } else if (xid.subdetId()==EcalEndcap) {
    
    EEDetId tempid(xid.rawId());

    int iSC = tempid.isc();
    int iX  = tempid.ix();
    int iY  = tempid.iy();
    int iZ  = tempid.zside();
    
    const int nSC = 312; 
    const int nEELM = 18;

    // Z+ side

    int indexSCpos[nSC] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163,164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 308, 309, 310, 311, 312, 313, 314, 315, 316 };

    int indexDCCpos[nSC] = { 48, 48, 48, 48, 48, 48, 48, 48, 47, 48, 48, 48, 48, 48, 48, 48, 48, 47, 47, 47, 48, 48, 48, 48, 48, 48, 48, 46, 47, 47, 47, 48, 48, 48, 48,48, 46, 47, 47, 47, 47, 48, 48, 48, 48, 46, 47, 47, 47, 47, 47, 47, 48, 48, 46, 47, 47, 47, 47, 47, 47, 47, 46, 47, 47, 47, 47, 47, 47, 46, 46, 47, 47, 47, 47, 46, 46, 47, 49, 49, 49, 49, 49, 49, 49, 49, 50, 49, 49, 49, 49, 49, 49, 49, 49, 50, 50, 50, 49, 49, 49, 49, 49, 49, 49, 51, 50, 50, 50, 49, 49, 49, 49, 49, 51, 50, 50, 50, 50, 49, 49, 49, 49, 51, 50, 50, 50, 50, 50, 50, 49, 49, 51, 50, 50, 50, 50, 50, 50, 50, 51, 50, 50, 50, 50, 50, 50, 51, 51, 50, 50, 50, 50, 51, 51, 50, 53, 53, 53, 53, 53, 53, 53, 53, 52, 52, 52, 53, 53, 53, 53, 53, 53, 51, 52, 52, 52, 52, 52, 52, 53, 53, 53, 51, 51, 52, 52, 52, 52, 52, 52, 52, 51, 51, 51, 52, 52, 52, 52, 52, 52, 51, 51, 51, 52, 52, 52, 52, 52, 52,51, 51, 51, 51, 52, 52, 52, 52, 51, 51, 51, 51, 52, 52, 52, 51, 51, 51, 51, 51, 52, 51, 51, 51, 53, 53, 53, 53, 53, 53, 53, 53, 54, 54, 54,53, 53, 53, 53, 53, 53, 46, 54, 54, 54, 54, 54, 54, 53, 53, 53, 46, 46, 54, 54, 54, 54, 54, 54, 54, 46, 46, 46, 54, 54, 54, 54, 54, 54, 46, 46, 46, 54, 54, 54, 54, 54, 54, 46, 46, 46, 46, 54, 54, 54, 54, 46, 46, 46, 46, 54, 54, 54, 46, 46, 46, 46, 46, 54, 46, 46, 46 };

    // Z- side

    int indexSCneg[nSC] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73,74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 308, 309, 310, 311, 312, 313, 314, 315, 316 };

    int indexDCCneg[nSC] = { 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 1, 2, 2, 2, 3, 3, 3, 3, 3, 1, 2, 2, 2, 2, 3, 3, 3, 3, 1, 2, 2, 2, 2, 2, 2, 3, 3, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 6, 5, 5, 5, 4, 4, 4, 4, 4, 6, 5, 5, 5, 5, 4, 4, 4, 4, 6, 5, 5, 5, 5, 5, 5, 4, 4, 6, 5, 5, 5, 5, 5, 5, 5, 6, 5, 5, 5, 5, 5, 5, 6, 6, 5, 5, 5, 5, 6, 6, 5, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 8, 8, 8, 8, 8, 8, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 6, 6, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 7, 7, 7, 7, 7, 7, 6, 6, 6, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 7, 7, 7, 7, 6, 6, 6, 6, 7, 7, 7, 6, 6, 6, 6, 6,7, 6, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 8, 8, 8, 8, 8, 8, 1, 9, 9, 9, 9, 9, 9, 8, 8, 8, 1, 1, 9, 9, 9, 9, 9, 9, 9, 1, 1, 1, 9, 9, 9, 9, 9, 9, 1, 1, 1, 9, 9, 9, 9, 9, 9, 1, 1, 1, 1, 9, 9, 9, 9, 1, 1, 1, 1, 9, 9, 9, 1, 1, 1, 1, 1, 9, 1, 1, 1 };

    int numDCC[nEELM] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 46, 47, 48, 49, 50, 51, 52, 53, 54 };
    int numLM[nEELM] = { 73, 74, 75, 76, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89, 90, 92 };

    int tempdcc = 0;

    // assign SC to DCC
    if (iZ>0) {
      for (int i=0; i<nSC; i++) {
	if (indexSCpos[i]==iSC) {
	  tempdcc = indexDCCpos[i];
	  break;
	}      
      }
    } else {
      for (int i=0; i<nSC; i++) {
	if (indexSCneg[i]==iSC) {
	  tempdcc = indexDCCneg[i];
	  break;
	}      
      }
    }
    
    // now assign LM number based on DCC
    for (int j=0; j<nEELM; j++) {
      if (tempdcc==numDCC[j]) {
	iLM = numLM[j];
	break;
      } 
    }
    
    // now assign LM readout number for two exceptional cases:
    if (tempdcc==53&&iX>50) {
      iLM++;
    } else if (tempdcc==8&&iX>50) {
      iLM++;
    }

    //    std::cout << "DCC , LM ---> " << tempdcc << " " << iLM << std::endl;
    std::cout << "SC DCC : LM : XYZ ---> " << iSC << " \t" << tempdcc << ": \t" << iLM << ": \t" << iX << " \t" << iY << " \t" << iZ << std::endl;
  
    //    iLM = tempid.isc();
  } else {
    edm::LogError("EcalLaserDbService") << " getLMNumber: DetId is not in ECAL." << endl;

  }

  return iLM;

}


EVENTSETUP_DATA_REG(EcalLaserDbService);
