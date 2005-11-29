#include "CalibFormats/SiStripObjects/interface/SiStripControlConnectivity.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include <iostream>
using namespace std;

//
// -- Given a FedReference returns the connected DetUnitId
//
DetId SiStripControlConnectivity::getDetId(FedReference& fed_ref){
  SiStripControlConnectivity::MapType::iterator CPos = theMap.find(fed_ref);
  if (CPos != theMap.end()) return (CPos->second).first;
  return DetId(0);
}
//
// -- Given Fed id and fed channel  returns the connected DetUnitId
//
DetId SiStripControlConnectivity::getDetId(unsigned short fed_id, unsigned short fed_channel){
  SiStripControlConnectivity::MapType::iterator CPos = 
             theMap.find(SiStripControlConnectivity::FedReference(fed_id, fed_channel));
  if (CPos != theMap.end()) return (CPos->second).first;
  return DetId(0);
}
//
//  get Connected DetUnit Ids in a given Fed Number
//
int SiStripControlConnectivity::getDetIds(unsigned short fed_num, 
                     unsigned short max_channels, vector<DetId>& det_ids){
  for (unsigned short ich = 0; ich < max_channels; ich++) {
    SiStripControlConnectivity::MapType::iterator CPos = theMap.find(FedReference(fed_num, ich));
    if (CPos != theMap.end()) {
       DetId idet = (CPos->second).first;
      unsigned short ipair = (CPos->second).second;
      if (ipair == 0) det_ids.push_back(idet);
    }
  }
  return det_ids.size();
}
//
// -- Given a DetId returns the list of corresponding FedChannels 
//
unsigned short SiStripControlConnectivity::getFedIdAndChannels(DetId id,
                                    vector<unsigned short>& fedChannels){
  unsigned short fed_id = 0;
  for (SiStripControlConnectivity::MapType::iterator it = theMap.begin() ; it != theMap.end() ; it++){
    if (it == theMap.begin()) fed_id = (it->first).first;
    if (((*it).second).first == id) fedChannels.push_back((it->first).second);
  }
  return fed_id;
}
//
// -- Given a Fed Reference returns the Apv Pair # of a GeomDetUnit connected
//
unsigned short SiStripControlConnectivity::getPairNumber(SiStripControlConnectivity::FedReference& fed_ref){
  unsigned int ipair = 10;
  SiStripControlConnectivity::MapType::iterator CPos = theMap.find(fed_ref);
  if (CPos != theMap.end()) ipair = (CPos->second).second;
  return ipair;
}
//
// -- Given a Fed Reference returns the Apv Pair# of a GeomDetUnit connected
//
unsigned short SiStripControlConnectivity::getPairNumber(unsigned short fed_id, unsigned short fed_channel){
  unsigned int ipair = 10;
  SiStripControlConnectivity::MapType::iterator CPos = 
            theMap.find(SiStripControlConnectivity::FedReference(fed_id,fed_channel));
  if (CPos != theMap.end()) ipair = (CPos->second).second;
  return ipair;
}
//
// -- Given a Fed Reference returns the DetPair
//
void SiStripControlConnectivity::getDetPair(SiStripControlConnectivity::FedReference& fed_ref, DetPair& det_pair){
  SiStripControlConnectivity::MapType::iterator CPos = theMap.find(fed_ref);
  if (CPos != theMap.end()) {
    det_pair = CPos->second;
  } else  {
    cout << " COULD NOT FIND DetPair ..." << endl;
  }
}
//
// -- Given a Fed id and channel number returns the DetPair (R.B)
//
void SiStripControlConnectivity::getDetPair(unsigned short fed_id, unsigned short fed_channel, DetPair& det_pair){

 if ( fed_id < detUnitMap_.size() ) {
    if ( fed_channel < detUnitMap_[fed_id].size() ) {
      det_pair = detUnitMap_[fed_id][fed_channel];
    } else {
      cout << "FedToDetUnitMapper::getDetUnit(...) : " 
	   << "ERROR : \"fed_channel > detUnitMap_[fed_id].size()\": "
	   << fed_channel << " > " << detUnitMap_[fed_id].size();
    
    }
  } else {
    cout << "FedToDetUnitMapper::getDetUnit(...) : " 
	 << "ERROR : \"fed_id > detUnitMap_.size()\": " 
	 << fed_id << " > " << detUnitMap_.size();
  }

}
//
// -- Given a FedChannel and its strip number returns exact strip 
//    number of the connected module
//
int SiStripControlConnectivity::getStripNumber(SiStripControlConnectivity::FedReference& fed_ref, int strip){
  int pair = getPairNumber(fed_ref);
  if (pair == -1){
    cout <<" ERROR!!!!! "<<endl;
    abort();
  }
  return strip + (256*pair);
}
//
// -- Set the pair of a FedChannel and corresponding DetPair
//
void SiStripControlConnectivity::setPair(FedReference& fed_ref, DetPair& dp){
  theMap[fed_ref] = dp;

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
void SiStripControlConnectivity::debug (){
  //
  // Loop over the map and print
  //
  for (SiStripControlConnectivity::MapType::iterator it = theMap.begin(); 
                               it!=theMap.end() ; it++){
    cout <<" Fed # "<< (it->first).first  <<" " 
	 <<" Ch # " << (it->first).second <<" attached to Det Id "
	 <<(*it).second.first.rawId() <<" in position "
         <<(*it).second.second << endl;
  }
}
//
// Get the whole Map
//
const SiStripControlConnectivity::MapType& SiStripControlConnectivity::getFedList(){
  return theMap;
}
//
// Get list of FED numbers in a vector
//
void SiStripControlConnectivity::getConnectedFedNumbers(vector<unsigned short>& feds) {
  feds.clear();
  unsigned short last_element = 99999;
  for (SiStripControlConnectivity::MapType::iterator it = theMap.begin(); 
                                 it!=theMap.end(); it++){
    unsigned short fed_id = (it->first).first;
    if ((it->second).first.rawId() != 0 && last_element != fed_id) feds.push_back(fed_id);
    if (last_element == 9999 || last_element != fed_id) last_element = fed_id;
  }
}
//
// Get a map of FED numbers and connected detids 
//
void SiStripControlConnectivity::getDetPartitions(map<unsigned short, vector<DetId> >& partitions){
  partitions.clear();
  unsigned short last_element = 99999;
  for (SiStripControlConnectivity::MapType::iterator it = theMap.begin(); 
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
}


EVENTSETUP_DATA_REG(SiStripControlConnectivity);
