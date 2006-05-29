#include <cmath>
#include <iostream>
#include "L1Trigger/RPCTrigger/src/RPCDetInfo.h"

//
//
//
// TODO: check given data
RPCDetInfo::RPCDetInfo(uint32_t DetId, int region, int ring, int station, int layer, int roll){

    
  mDetId = DetId;
  mRegion = region; 
  mRing = ring;
  mStation = station;
  mLayer = layer;
  mRoll = roll;
  mHwPlane = getHwPlane();
      


}

//
// gets coresponding curlid.
// 
int RPCDetInfo::getCurlId(){

  //int towerMin = etaToTower(mEtaMin);
  //int towerMax = etaToTower(mEtaMax);
  //int signMin = etaToSign(mEtaMin);
  //int signMax = etaToSign(mEtaMax);

  // Constants are added to have positive numbers
  
  //std::cout << mHwPlane << " " << mRegion+2 << " " << mRing+2 << " " << mRoll << std::endl;
      
  int curlId = 1000*(mRegion+2) +  // barell is now 2, endcaps are 1 and 3
                100*(mRing+2) +    // barell may have negative wheel no !                
                 10*mHwPlane +     //1...6
                  1*(mRoll);
                   

  return curlId;
}

//
// Converts eta to coresponding tower number
// TODO: store somewhere MAXTOWER no.
// TODO: store somewhere the max eta value (it will tell us if we properly used geometry)
//
int RPCDetInfo::etaToTower(float eta){

  int sign = etaToSign(eta);
  eta = std::fabs(eta);

  
  if ( eta  > 2.15 ) {  // the number is arbitrary but close to real world limit (2.1),
                        // tests consistency of data
    RPCDetId tmpDetId(mDetId);
    std::cout << "Trouble with detId " << mDetId
              << " eta=" << eta
              << " region= " << tmpDetId.region()
              << std::endl;
  }

  int tower = 0;
  // The highest tower no is 16
  while ( (eta > mTowerBounds[tower]) && (tower!=16) ){
    tower++;
  }


  //std::cout << "eta " << eta << " tower " << tower << std::endl;

  if (sign == 0)
    return -tower;
  else
    return tower;

}

//
// sets hardware plane number (mHwPlane)
// TODO: check layer convention (which is inner/outer)
//       will show up with number of curls beeing reference
//int RPCDetInfo::getHwPlane(int region = mRegion, int station = mStation, int layer = mLayer){
int RPCDetInfo::getHwPlane()
{    
  int region = mRegion;
  int station = mStation;
  int layer = mLayer;
    
  int hwPlane = 0;
  if (region != 0){ // endcaps
    hwPlane = station;
  }
  // Now comes the barell
  else if ( station > 2 ){
    hwPlane = station+2;
  } 
  else if ( station == 1 && layer == 1) {
    hwPlane = 1;
  }
  else if ( station == 1 && layer == 2) {
    hwPlane = 2;
  }
  else if ( station == 2 && layer == 1) {
    hwPlane = 3;
  }
  else if ( station == 2 && layer == 2) {
    hwPlane = 4;
  }

  return hwPlane;
  
}

//
// Gives sign of eta (+) -> 1; (-) -> 0
//
int RPCDetInfo::etaToSign(float eta){

  if (eta < 0) return 0;
  return 1;

}
uint32_t RPCDetInfo::rawId(){
  return mDetId;
}
const float RPCDetInfo::mTowerBounds[] = {0.07, 0.27, 0.44, 0.58, 0.72, 0.83, 0.93, 1.04, 1.14,
                            1.24, 1.36, 1.48, 1.61, 1.73, 1.85, 1.97, 2.10 };
