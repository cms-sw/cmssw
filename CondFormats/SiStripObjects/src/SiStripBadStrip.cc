#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"

#include <algorithm>

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

const SiStripBadStrip::Range SiStripBadStrip::getRange(const uint32_t DetId) const {
  // get SiStripBadStrip Range of DetId
  
  RegistryIterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiStripBadStrip::StrictWeakOrdering());
  if (p==indexes.end()|| p->detid!=DetId) 
    return SiStripBadStrip::Range(v_badstrips.end(),v_badstrips.end()); 
  else {
    __builtin_prefetch((&v_badstrips.front())+p->ibegin);
    __builtin_prefetch((&v_badstrips.front())+p->ibegin+24);
    __builtin_prefetch((&v_badstrips.front())+p->iend-24);
    return SiStripBadStrip::Range(v_badstrips.begin()+p->ibegin,v_badstrips.begin()+p->iend);
  }
}

SiStripBadStrip::Range SiStripBadStrip::getRangeByPos(unsigned short pos) const {
  if (pos>indexes.size()) return Range(v_badstrips.end(),v_badstrips.end()); 
  auto p = indexes.begin()+pos;
  __builtin_prefetch((&v_badstrips.front())+p->ibegin);
  __builtin_prefetch((&v_badstrips.front())+p->ibegin+24);
  __builtin_prefetch((&v_badstrips.front())+p->iend-24);
  return Range(v_badstrips.begin()+p->ibegin,v_badstrips.begin()+p->iend);
}


void SiStripBadStrip::getDetIds(std::vector<uint32_t>& DetIds_) const {
  // returns vector of DetIds in map
  SiStripBadStrip::RegistryIterator begin = indexes.begin();
  SiStripBadStrip::RegistryIterator end   = indexes.end();
  for (SiStripBadStrip::RegistryIterator p=begin; p != end; ++p) {
    DetIds_.push_back(p->detid);
  }
}

void SiStripBadStrip::printSummary(std::stringstream & ss) const {
  SiStripDetSummary summaryBadModules;
  SiStripDetSummary summaryBadStrips;

  // Loop on the vector<DetRegistry> and take the bad modules and bad strips
  Registry::const_iterator it = indexes.begin();
  for( ; it!=indexes.end(); ++it ) {
    summaryBadModules.add(it->detid);
    summaryBadStrips.add(it->iend - it->ibegin);
  }
  ss << "Summary of bad modules in detector:" << std::endl;
  summaryBadModules.print(ss, false);
  ss << "Summary of bad strip in detectors:" << std::endl;
  summaryBadStrips.print(ss, false);
}

void SiStripBadStrip::printDebug(std::stringstream & ss) const {
  ss << "Printing all bad strips for all DetIds" << std::endl;
  // Loop on the vector<DetRegistry> and take the bad modules and bad strips
  Registry::const_iterator it = indexes.begin();
  for( ; it!=indexes.end(); ++it ) {
//    ss << "For DetId = " << it->detid << std::endl;
    SiStripBadStrip::Range range(getRange(it->detid));
    for( std::vector<unsigned int>::const_iterator badStrip = range.first;
         badStrip != range.second; ++badStrip ) {
      ss << "DetId="<<it->detid << " Strip=" << decode(*badStrip).firstStrip <<":"<<decode(*badStrip).range << " flag="<< decode(*badStrip).flag << std::endl;
    }
  }
}
