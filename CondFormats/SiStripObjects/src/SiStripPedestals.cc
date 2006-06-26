#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "FWCore/Utilities/interface/Exception.h"

bool SiStripPedestals::put(const uint32_t& DetId, Range input) {
  // put in SiStripPedestals of DetId

  Registry::iterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiStripPedestals::StrictWeakOrdering());
  if (p!=indexes.end() && p->detid==DetId)
    return false;
  
  size_t sd= input.second-input.first;
  DetRegistry detregistry;
  detregistry.detid=DetId;
  detregistry.ibegin=v_pedestals.size();
  detregistry.iend=v_pedestals.size()+sd;
  indexes.insert(p,detregistry);

  v_pedestals.insert(v_pedestals.end(),input.first,input.second);
  return true;
}

const SiStripPedestals::Range SiStripPedestals::getRange(const uint32_t& DetId) const {
  // get SiStripPedestals Range of DetId
  
  RegistryIterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiStripPedestals::StrictWeakOrdering());
  if (p==indexes.end()|| p->detid!=DetId) 
    return SiStripPedestals::Range(v_pedestals.end(),v_pedestals.end()); 
  else 
    return SiStripPedestals::Range(v_pedestals.begin()+p->ibegin,v_pedestals.begin()+p->iend);
}

void SiStripPedestals::getDetIds(std::vector<uint32_t>& DetIds_) const {
  // returns vector of DetIds in map
  SiStripPedestals::RegistryIterator begin = indexes.begin();
  SiStripPedestals::RegistryIterator end   = indexes.end();
  for (SiStripPedestals::RegistryIterator p=begin; p != end; ++p) {
    DetIds_.push_back(p->detid);
  }
}


void SiStripPedestals::setData(float ped, float lth, float hth, std::vector<char>& vped){
  
  unsigned int ped_  = (static_cast<unsigned int>(ped)) & 0xFFF; 
  unsigned int low_  = (static_cast<unsigned int>(lth*5.0+0.5)) & 0x3F; 
  unsigned int hig_  = (static_cast<unsigned int>(hth*5.0+0.5)) & 0x3F; 
  unsigned int data = (ped_ << 12) | (hig_ << 6) | low_ ;
  vped.resize(vped.size()+3);
  // insert in vector of char
  ::memcpy((void*)(&vped[vped.size()-3]),(void*)(&data),3);
}

float SiStripPedestals::getPed(const uint16_t& strip, const Range& range) const {
  if (strip>=(range.second-range.first)/3){
    throw cms::Exception("CorruptedData")
      << "[SiStripPedestals::getPed] looking for SiStripPedestals for a strip out of range: strip " << strip;
  }
  const DecodingStructure & s = (const DecodingStructure & ) *(range.first+strip*3);
  return (s.ped & 0x3FF);
}

float SiStripPedestals::getLowTh(const uint16_t& strip, const Range& range) const {
  if (strip>=(range.second-range.first)/3){
    throw cms::Exception("CorruptedData")
      << "[SiStripPedestals::getLowTh] looking for SiStripPedestals for a strip out of range: strip " << strip;
  }
  const DecodingStructure & s = (const DecodingStructure & ) *(range.first+strip*3);
  return (s.lth & 0x3F)/5.0;
}

float SiStripPedestals::getHighTh(const uint16_t& strip, const Range& range) const {
  if (strip>=(range.second-range.first)/3){
    throw cms::Exception("CorruptedData")
      << "[SiStripPedestals::getHighTh] looking for SiStripPedestals for a strip out of range: strip " << strip;
  }
  const DecodingStructure & s = (const DecodingStructure & ) *(range.first+strip*3);
  return (s.hth & 0x3F)/5.0;
}

