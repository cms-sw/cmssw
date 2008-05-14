#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

bool SiStripThreshold::put(const uint32_t& DetId, InputVector vect) {
  // put in SiStripThreshold::v_threshold of DetId
  Registry::iterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiStripThreshold::StrictWeakOrdering());
  if (p!=indexes.end() && p->detid==DetId){
    edm::LogError("SiStripThreshold") << "[" << __PRETTY_FUNCTION__ << "] SiStripThreshold for DetID " << DetId << " is already stored. Skippig this put" << std::endl;
    return false;
  }
  
  SiStripThreshold::Container::iterator new_end=compact(vect);

  size_t sd= new_end-vect.begin();
  DetRegistry detregistry;
  detregistry.detid=DetId;
  detregistry.ibegin=v_threshold.size();
  detregistry.iend=v_threshold.size()+sd;
  indexes.insert(p,detregistry);
  
  v_threshold.insert(v_threshold.end(),vect.begin(),new_end);
  
  return true;
}

SiStripThreshold::Container::iterator SiStripThreshold::compact(Container& input) {
  std::stable_sort(input.begin(),input.end());
  return std::unique(input.begin(),input.end());
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

void SiStripThreshold::setData(const uint16_t& strip, const float& lTh,const float& hTh, Container& vthr){
  Data a;
  a.encode(strip,lTh,hTh);
  vthr.push_back(a);
}

SiStripThreshold::Data SiStripThreshold::getData(const uint16_t& strip, const Range& range) const {
  uint16_t estrip=(strip & sistrip::FirstThStripMask_)<<sistrip::FirstThStripShift_ | (63 & sistrip::HighThStripMask_);
  ContainerIterator p = std::upper_bound(range.first,range.second,estrip,SiStripThreshold::dataStrictWeakOrdering());
  if (p!=range.first){
    return *(--p);
  }
  else{
    throw cms::Exception("CorruptedData")
      << "[SiStripThreshold::getData] asking for data for a strip " << strip << " lower then the first stored strip " << p->getFirstStrip();
  }
}
