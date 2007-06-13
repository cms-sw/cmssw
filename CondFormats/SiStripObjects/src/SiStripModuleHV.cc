#include "CondFormats/SiStripObjects/interface/SiStripModuleHV.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


bool SiStripModuleHV::put(std::vector<uint32_t>& DetId) {
  // put in SiStripModuelHV::v_hvoff of DetId
 
  
  v_hvoff.insert(v_hvoff.end(),DetId.begin(),DetId.end());
  std::sort(v_hvoff.begin(),v_hvoff.end());

  
  std::vector<uint32_t> v_detidcompare;
 
  for(int in= 0;in<v_hvoff.size();in++){
     v_detidcompare.push_back(v_hvoff[in]);
     
 
    if(in>0){
      if(v_detidcompare[in-1]==v_hvoff[in]){
	std::cout << "detid: " << v_hvoff[in] << "already stored, skipping this input \n";
	return false;}
    }
   
  }
    
  return true;
}


bool SiStripModuleHV::putalldetids(std::vector<uint32_t>& DetId) {
  // put all DetIds
 
  
  v_det_ids.insert(v_det_ids.end(),DetId.begin(),DetId.end());
  std::sort(v_det_ids.begin(),v_det_ids.end());

  
  std::vector<uint32_t> v_detidcompare;
 
  for(int in= 0;in<v_det_ids.size();in++){
     v_detidcompare.push_back(v_det_ids[in]);
     
 
    if(in>0){
      if(v_detidcompare[in-1]==v_det_ids[in]){
	std::cout << "detid: " << v_det_ids[in] << "already stored, skipping this input \n";
	return false;}
    }
   
  }
    
  return true;
}


void SiStripModuleHV::getDetIds(std::vector<uint32_t>& DetIds_) const {
  // returns vector of DetIds in map
  DetIds_.clear();
  DetIds_.insert(DetIds_.end(),v_hvoff.begin(),v_hvoff.end());
 }


void SiStripModuleHV::GetAllDetIds(std::vector<uint32_t>& AllDetIds) const {
  // returns a vector with all detids 
  AllDetIds.clear();
  AllDetIds.insert(AllDetIds.end(),v_det_ids.begin(),v_det_ids.end());
  

}



bool SiStripModuleHV::IsModuleHVOff(uint32_t DetID) const{
  return std::binary_search(v_hvoff.begin(),v_hvoff.end(),DetID);
}
