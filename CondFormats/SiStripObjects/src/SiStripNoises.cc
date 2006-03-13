#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"

bool SiStripNoises::put(const uint32_t& DetId, Range input) {
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

const SiStripNoises::Range SiStripNoises::getRange(const uint32_t& DetId) const {
  // get SiStripNoises Range of DetId
  
  RegistryIterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,StrictWeakOrdering());
  if (p==indexes.end()|| p->detid!=DetId) 
    return SiStripNoises::Range(v_noises.end(),v_noises.end()); 
  else 
    return SiStripNoises::Range(v_noises.begin()+p->ibegin,v_noises.begin()+p->iend);
}

void SiStripNoises::getDetIds(std::vector<uint32_t>& DetIds_) const {
  // returns vector of DetIds in map
  SiStripNoises::RegistryIterator begin = indexes.begin();
  SiStripNoises::RegistryIterator end   = indexes.end();
  for (; begin != end; ++begin) {
    DetIds_.push_back(begin->detid);
  }
}



