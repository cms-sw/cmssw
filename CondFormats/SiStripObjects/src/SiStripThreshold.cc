#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

bool SiStripThreshold::put(const uint32_t& DetId, Range input) {
  // put in SiStripThreshold::v_threshold of DetId
  Registry::iterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiStripThreshold::StrictWeakOrdering());
  if (p!=indexes.end() && p->detid==DetId){
    edm::LogError("SiStripThreshold") << "[" << __PRETTY_FUNCTION__ << "] SiStripThreshold for DetID " << DetId << " is already stored. Skippig this put" << std::endl;
    return false;
  }
  
  size_t sd= input.second-input.first;
  DetRegistry detregistry;
  detregistry.detid=DetId;
  detregistry.ibegin=v_threshold.size();
  detregistry.iend=v_threshold.size()+sd;
  indexes.insert(p,detregistry);

  v_threshold.insert(v_threshold.end(),input.first,input.second);
  return true;
}

const SiStripThreshold::Range SiStripThreshold::getRange(const uint32_t& DetId) const {
  // get SiStripThreshold Range of DetId
  
  RegistryIterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiStripThreshold::StrictWeakOrdering());
  if (p==indexes.end()|| p->detid!=DetId) 
    return SiStripThreshold::Range(v_threshold.end(),v_threshold.end()); 
  else 
    return SiStripThreshold::Range(v_threshold.begin()+p->ibegin,v_threshold.begin()+p->iend);
}


void SiStripThreshold::getDetIds(std::vector<uint32_t>& DetIds_) const {
  // returns vector of DetIds in map
  SiStripThreshold::RegistryIterator begin = indexes.begin();
  SiStripThreshold::RegistryIterator end   = indexes.end();
  for (SiStripThreshold::RegistryIterator p=begin; p != end; ++p) {
    DetIds_.push_back(p->detid);
  }
}

void SiStripThreshold::setData(float lth, float hth, std::vector<unsigned int>& vthr){
  
  //unsigned int ped_  = (static_cast<unsigned int>(ped)) & 0xFFF; 
  unsigned int low_  = (static_cast<unsigned int>(lth*5.0+0.5)) & 0x3F; 
  unsigned int hig_  = (static_cast<unsigned int>(hth*5.0+0.5)) & 0x3F; 
  unsigned int data = /*(ped_ << 12)|*/ (hig_ << 6) | low_ ;
  vthr.resize(vthr.size()+4);
  // insert in vector of char
  ::memcpy((void*)(&vthr[vthr.size()-4]),(void*)(&data),2);
}
/*
float SiStripThreshold::getLowTh(const uint16_t& strip, const Range& range) const {
  if (strip>=(range.second-range.first)/3){
    throw cms::Exception("CorruptedData")
      << "[SiStripPedestals::getLowTh] looking for SiStripPedestals for a strip out of range: strip " << strip;
  }
  //const DecodingStructure & s = (const DecodingStructure & ) *(range.first+strip*3);
  //return (s.lth & 0x3F)/5.0;
	return static_cast<float> (decode(strip,range));
}

float SiStripThreshold::getHighTh(const uint16_t& strip, const Range& range) const {
  if (strip>=(range.second-range.first)/3){
    throw cms::Exception("CorruptedData")
      << "[SiStripPedestals::getHighTh] looking for SiStripPedestals for a strip out of range: strip " << strip;
  }
  //const DecodingStructure & s = (const DecodingStructure & ) *(range.first+strip*3);
  //return (s.hth & 0x3F)/5.0;
	return static_cast<float> (decode(strip,range));
}*/
