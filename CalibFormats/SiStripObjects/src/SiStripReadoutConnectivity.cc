#include "CalibFormats/SiStripObjects/interface/SiStripReadoutConnectivity.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include <iostream>
using namespace std;

//
// -- Given a FedReference returns the connected DetUnitId
// 
DetId SiStripReadoutConnectivity::getDetId(FedReference& fed_ref){
  return getDetId(fed_ref.first, fed_ref.second);
}
//
// -- Given Fed id and fed channel  returns the connected DetUnitId
//
DetId SiStripReadoutConnectivity::getDetId(unsigned short fed_id, unsigned short fed_channel){
  DetPair det_pair;
  getDetPair(fed_id, fed_channel, det_pair);
  if (det_pair.first.null()) return DetId(0);
  else return det_pair.first; 
}
//
//  get Connected DetUnit Ids in a given Fed Number
//
int SiStripReadoutConnectivity::getDetIds(unsigned short fed_num, 
                     unsigned short max_channels, vector<DetId>& det_ids){
  det_ids.clear();
  for (unsigned short ich = 0; ich < max_channels; ich++) {
    DetPair det_pair;
    getDetPair(fed_num, ich, det_pair);
    if (!det_pair.first.null() && det_pair.second == 0) 
                                 det_ids.push_back(det_pair.first);
  }
  return det_ids.size();
}
//
// -- Given a DetId returns the list of corresponding FedChannels 
//
unsigned short SiStripReadoutConnectivity::getFedIdAndChannels(DetId id,
                           map<unsigned short, unsigned short>& fedChannels){
  fedChannels.clear();
  unsigned short fed_id = 999999;
  for (unsigned short i = 0; i < detUnitMap_.size(); i++) {
    for(unsigned short j = 0; j < detUnitMap_[i].size(); j++) {
      if (detUnitMap_[i][j].first == id) {
        fed_id = i;
        unsigned short ipair = detUnitMap_[i][j].second;
        fedChannels[ipair] = j;
      }
    }
    if (fed_id != 999999) break;
  }
  return fed_id;
}
//
// -- Given a Fed Reference returns the Apv Pair# of a GeomDetUnit connected
//
unsigned short SiStripReadoutConnectivity::getPairNumber(unsigned short fed_id, unsigned short fed_channel){
  DetPair det_pair;
  getDetPair(fed_id, fed_channel, det_pair);
  if (det_pair.first.null()) return 99;
  else return det_pair.second; 
}
//
// -- Given a Fed Reference returns the Apv Pair # of a GeomDetUnit connected
//
unsigned short SiStripReadoutConnectivity::getPairNumber(SiStripReadoutConnectivity::FedReference& fed_ref){
  return getPairNumber(fed_ref.first, fed_ref.second);
}
//
// -- Given a Fed Reference returns the DetPair
//
void SiStripReadoutConnectivity::getDetPair(SiStripReadoutConnectivity::FedReference& fed_ref, DetPair& det_pair){
  getDetPair(fed_ref.first, fed_ref.second, det_pair);
}
//
// -- Given a Fed id and channel number returns the DetPair (R.B)
//
void SiStripReadoutConnectivity::getDetPair(unsigned short fed_id, unsigned short fed_channel, DetPair& det_pair){

 if ( fed_id < detUnitMap_.size() ) {
    if ( fed_channel < detUnitMap_[fed_id].size() ) {
      det_pair = detUnitMap_[fed_id][fed_channel];
    } else {
      cout << "FedToDetUnitMapper::getDetUnit(...) : " 
	   << "ERROR : \"fed_channel > detUnitMap_[fed_id].size()\": "
	   << fed_channel << " > " << detUnitMap_[fed_id].size();
      det_pair = DetPair(DetId(0),99);
    }
  } else {
    cout << "FedToDetUnitMapper::getDetUnit(...) : " 
	 << "ERROR : \"fed_id > detUnitMap_.size()\": " 
	 << fed_id << " > " << detUnitMap_.size();
    det_pair = DetPair(DetId(0),99);
  }

}
//
// -- Given a FedChannel and its strip number returns exact strip 
//    number of the connected module
//
int SiStripReadoutConnectivity::getStripNumber(SiStripReadoutConnectivity::FedReference& fed_ref, int strip){
  int pair = getPairNumber(fed_ref);
  if (pair == 99){
    cout <<" ERROR!!!!! "<<endl;
    abort();
  }
  return strip + (256*pair);
}
//
// -- Set the pair of a FedChannel and corresponding DetPair
//
void SiStripReadoutConnectivity::setPair(FedReference& fed_ref, DetPair& dp){

  // below: fills two-dimensional array of DetPair's (R.B)
  int fed_id = fed_ref.first;
  int fed_channel = fed_ref.second;

  // Check FED id 
  if ( fed_id < static_cast<int>(0) || 
       fed_id > static_cast<int>(1023) ) { 
    cout << "[FedToDetUnitMapper::setPair] " 
	 << "ERROR : Unexpected FED id: " << fed_id;
  }
  // check FED channel
  if ( fed_channel < static_cast<int>(0) || 
       fed_channel > static_cast<int>(95) ) { 
    cout << "[FedToDetUnitMapper::setPair] " 
	 << "ERROR : Unexpected FED channel: " << fed_channel;
  } 
  
  // Increase size of (fed_id, first dim) vector if necessary
  if ( fed_id >= static_cast<int>(detUnitMap_.size()) ) { 
    detUnitMap_.reserve(fed_id+1); 
    detUnitMap_.resize( fed_id+1, vector<DetPair>( 96, DetPair(DetId(0),0) ) ); 
  }

  // Increase size of (fed_channel, second dim) vector if necessary
  if ( fed_channel >= static_cast<int>(detUnitMap_[fed_id].size()) ) { 
    detUnitMap_[fed_id].reserve(fed_channel+1); 
    detUnitMap_[fed_id].resize( fed_channel+1, DetPair(DetId(0),0) ); 
  }

  // Store DetPair info in vector
  DetId det = dp.first;
  int iapv = dp.second;
  detUnitMap_[fed_id][fed_channel] = DetPair( det, iapv );

}
//
// -- Debug
//
void SiStripReadoutConnectivity::debug (){
  //
  // Loop over all the FED channels  and print
  //
  for (unsigned short i = 0; i < detUnitMap_.size(); i++) {
    for(unsigned short j = 0; j < detUnitMap_[i].size(); j++) {
      if (detUnitMap_[i][j].first.null()) continue;
      cout <<" Fed # "<< i  <<" " 
	 <<" Ch # " << j <<" attached to Det Id "
	 << detUnitMap_[i][j].first.rawId() <<" in position "
         << detUnitMap_[i][j].second << endl;
    }
  }
}
//
// Get list of FED numbers in a vector
//
void SiStripReadoutConnectivity::getConnectedFedNumbers(vector<unsigned short>& feds) {
  feds.clear();
  unsigned short last_element = 99999;
  for (unsigned short i = 0; i < detUnitMap_.size(); i++) {
    for(unsigned short j = 0; j < detUnitMap_[i].size(); j++) {
      if (detUnitMap_[i][j].first.null()) continue;
      else if (last_element == 99999 || last_element != i) {
        feds.push_back(i);
        last_element =i; 
      }
    }
  }
}
//
// Get a map of FED numbers and connected detids 
//
/*void SiStripReadoutConnectivity::getDetPartitions(map<unsigned short, vector<DetId> >& partitions){
  partitions.clear();
  unsigned short last_element = 99999;
  for (SiStripReadoutConnectivity::MapType::iterator it = theMap.begin(); 
                    it!=theMap.end() ; it++){
    
    unsigned short fed_id = (it->first).first;
    if ((it->second).first.rawId() != 0 && last_element != fed_id) {
      vector<DetId> det_ids;
      int nDets = getDetIds(fed_id, 96, det_ids);
      if (nDets != 0)
         partitions.insert(pair<unsigned short, vector<DetId> >(fed_id,det_ids));
    }
    if (last_element == 9999 || last_element != fed_id) last_element = fed_id;    
  }
}*/


EVENTSETUP_DATA_REG(SiStripReadoutConnectivity);
