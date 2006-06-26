#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "FWCore/Utilities/interface/Exception.h"

bool SiStripNoises::put(const uint32_t& DetId, Range input) {
  // put in SiStripNoises of DetId

  Registry::iterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiStripNoises::StrictWeakOrdering());
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

const SiStripNoises::Range SiStripNoises::getRange(const uint32_t& DetId) const {
  // get SiStripNoises Range of DetId
  
  RegistryIterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiStripNoises::StrictWeakOrdering());
  if (p==indexes.end()|| p->detid!=DetId) 
    return SiStripNoises::Range(v_noises.end(),v_noises.end()); 
  else 
    return SiStripNoises::Range(v_noises.begin()+p->ibegin,v_noises.begin()+p->iend);
}

void SiStripNoises::getDetIds(std::vector<uint32_t>& DetIds_) const {
  // returns vector of DetIds in map
  SiStripNoises::RegistryIterator begin = indexes.begin();
  SiStripNoises::RegistryIterator end   = indexes.end();
  for (SiStripNoises::RegistryIterator p=begin; p != end; ++p) {
    DetIds_.push_back(p->detid);
  }
}

float SiStripNoises::getNoise(const uint16_t& strip, const Range& range) const {
  if (strip>=range.second-range.first){
    throw cms::Exception("CorruptedData")
      << "[SiStripNoises::getNoise] looking for SiStripNoises for a strip out of range: strip " << strip;
  }

  return static_cast<float> (abs(*(range.first+strip))/10.0);
}

bool SiStripNoises::getDisable(const uint16_t& strip, const Range& range) const {
  if (strip>=range.second-range.first){
    throw cms::Exception("CorruptedData")
      << "[SiStripNoises::getDisable] looking for SiStripNoises for a strip out of range: strip " << strip;
  }
  return *(range.first+strip) > 0 ? false : true ;
}


void SiStripNoises::setData(float noise_, bool disable_, std::vector<short>& vped){
  vped.push_back(( disable_ ? -1 : 1 ) *  (static_cast<int16_t>  (noise_*10.0 + 0.5) & 0x01FF)) ;
}
