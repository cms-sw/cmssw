#include "CalibFormats/SiStripObjects/interface/SiStripControlConnectivity.h"

#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include <iostream>
#include <sstream>
using namespace std;
//
// -- Given Control References returns the Corresponding DetUnitId
//
uint32_t SiStripControlConnectivity::getDetId (short fec,short ring,short ccu,short i2c){
  for (SiStripControlConnectivity::MapType::iterator it = theMap.begin();
       it != theMap.end(); it++) {
    if (it->second.fecNumber  == fec &&
        it->second.ringNumber == ring &&
        it->second.ccuAddress == ccu &&
        it->second.i2cChannel == i2c) return it->first;
  }
  return 0;
}
//
// Get Connection Information
//
void SiStripControlConnectivity::getConnectionInfo(uint32_t det_id, 
                              short& fec,short& ring,short& ccu,short& i2c){
  SiStripControlConnectivity::MapType::iterator CPos = theMap.find(det_id);
  if (CPos != theMap.end()) {
    fec  = CPos->second.fecNumber;
    ring = CPos->second.ringNumber;
    ccu  = CPos->second.ccuAddress;
    i2c  = CPos->second.i2cChannel;
  }
}
//
//  Set Pair
//
void SiStripControlConnectivity::setPair(uint32_t det_id,short& fec, 
    short& ring,short& ccu,short& i2c, vector<short>& apvs){
  SiStripControlConnectivity::DetControlInfo detInfo;
  detInfo.fecNumber  = fec;
  detInfo.ringNumber = ring;
  detInfo.ccuAddress = fec;
  detInfo.i2cChannel = fec;
  detInfo.apvAddresses = apvs;

  theMap[det_id] = detInfo;

}
//
// Get Detector IDs for a given set of Fec#, Ring# and CCU#
//
int SiStripControlConnectivity::getDetIds(short fec, short ring, 
					  short ccu, vector<uint32_t>& dets) {
  dets.clear();
  for (SiStripControlConnectivity::MapType::iterator it = theMap.begin();
       it != theMap.end(); it++) {
    if (it->second.fecNumber  == fec &&
        it->second.ringNumber == ring &&
        it->second.ccuAddress == ccu ) dets.push_back(it->first);
  }
  return dets.size();
}
//
// Get Detector IDs for a given set of Fec#, Ring# 
//
int SiStripControlConnectivity::getDetIds(short fec, short ring, vector<uint32_t>& dets) {
  dets.clear();
  for (SiStripControlConnectivity::MapType::iterator it = theMap.begin();
       it != theMap.end(); it++) {
    if (it->second.fecNumber  == fec &&
        it->second.ringNumber == ring) dets.push_back(it->first);
  }
  return dets.size();
}
//
// Get Detector IDs for a given set of Fec#, Ring# 
//
int SiStripControlConnectivity::getDetIds(short fec, vector<uint32_t>& dets) {
  dets.clear();
  for (SiStripControlConnectivity::MapType::iterator it = theMap.begin();
       it != theMap.end(); it++) {
    if (it->second.fecNumber  == fec) dets.push_back(it->first);
  }
    return dets.size();
}

EVENTSETUP_DATA_REG(SiStripControlConnectivity);
