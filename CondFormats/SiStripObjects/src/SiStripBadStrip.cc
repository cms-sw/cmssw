#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

bool SiStripBadStrip::put(const uint32_t& DetId, Range input) {
  // put in SiStripBadStrip::v_badstrips of DetId
  Registry::iterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiStripBadStrip::StrictWeakOrdering());
  if (p!=indexes.end() && p->detid==DetId){
    edm::LogError("SiStripBadStrip") << "[" << __PRETTY_FUNCTION__ << "] SiStripBadStrip for DetID " << DetId << " is already stored. Skippig this put" << std::endl;
    return false;
  }
  
  size_t sd= input.second-input.first;
  DetRegistry detregistry;
  detregistry.detid=DetId;
  detregistry.ibegin=v_badstrips.size();
  detregistry.iend=v_badstrips.size()+sd;
  indexes.insert(p,detregistry);

  v_badstrips.insert(v_badstrips.end(),input.first,input.second);
  return true;
}

const SiStripBadStrip::Range SiStripBadStrip::getRange(const uint32_t& DetId) const {
  // get SiStripBadStrip Range of DetId
  
  RegistryIterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiStripBadStrip::StrictWeakOrdering());
  if (p==indexes.end()|| p->detid!=DetId) 
    return SiStripBadStrip::Range(v_badstrips.end(),v_badstrips.end()); 
  else 
    return SiStripBadStrip::Range(v_badstrips.begin()+p->ibegin,v_badstrips.begin()+p->iend);
}


void SiStripBadStrip::getDetIds(std::vector<uint32_t>& DetIds_) const {
  // returns vector of DetIds in map
  SiStripBadStrip::RegistryIterator begin = indexes.begin();
  SiStripBadStrip::RegistryIterator end   = indexes.end();
  for (SiStripBadStrip::RegistryIterator p=begin; p != end; ++p) {
    DetIds_.push_back(p->detid);
  }
}

 int SiStripBadStrip::getBadStrips(const Range& range) const {
   return (*range.first);
 }

