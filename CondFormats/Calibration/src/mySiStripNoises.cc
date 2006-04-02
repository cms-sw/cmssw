#include "CondFormats/Calibration/interface/mySiStripNoises.h"
#include <algorithm>
float mySiStripNoises::SiStripData::getNoise() const{
  return static_cast<float>(abs(mySiStripNoises::SiStripData::Data)/10.0);
} 
bool mySiStripNoises::SiStripData::getDisable() const{
  return ( (mySiStripNoises::SiStripData::Data>=0) ? false : true );
}
void mySiStripNoises::SiStripData::setData(short data){
  mySiStripNoises::SiStripData::Data=data ;
}
void mySiStripNoises::SiStripData::setData(float noisevalue,bool disable){
  short noise =  static_cast<short>  (noisevalue*10.0 + 0.5) & 0x01FF;
  mySiStripNoises::SiStripData::Data = ( disable ? -1 : 1 ) * noise;
}
bool mySiStripNoises::put(const uint32_t& DetId, Range input) {
// put in SiStripNoises of DetId

  Registry::iterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,StrictWeakOrdering());
  if (p!=indexes.end() && p->detid==DetId)
    return false;
  
  size_t sd= input.second-input.first;
  DetRegistry detregistry;
  detregistry.detid=DetId;
  detregistry.ibegin=v_noises.size();
  detregistry.iend=v_noises.size()+sd;
  indexes.insert(p,detregistry);

  v_noises.insert(v_noises.end(),input.first,input.second);
  return true;
}

const mySiStripNoises::Range mySiStripNoises::getRange(const uint32_t& DetId) const {
  // get SiStripNoises Range of DetId
  RegistryIterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,StrictWeakOrdering());
  if (p==indexes.end()|| p->detid!=DetId) 
    return mySiStripNoises::Range(v_noises.end(),v_noises.end()); 
  else 
    return mySiStripNoises::Range(v_noises.begin()+p->ibegin,v_noises.begin()+p->iend);
}

void mySiStripNoises::getDetIds(std::vector<uint32_t>& DetIds) const {
  // returns vector of DetIds in map
  mySiStripNoises::RegistryIterator begin = indexes.begin();
  mySiStripNoises::RegistryIterator end   = indexes.end();
  for (mySiStripNoises::RegistryIterator p=begin; p != end; ++p) {
    DetIds.push_back(p->detid);
  }
}



