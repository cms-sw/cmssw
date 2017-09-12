#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"

#include <algorithm>

bool SiStripApvGain::put(const uint32_t& DetId, Range input) {
  // put in SiStripApvGain of DetId
  RegistryIterator p = std::lower_bound(v_detids.begin(),v_detids.end(),DetId);
  if (p!=v_detids.end() && *p==DetId){
    edm::LogError("SiStripApvGain") << "[" << __PRETTY_FUNCTION__ << "] SiStripApvGain for DetID " << DetId << " is already stored. Skippig this put" << std::endl;
    return false;
  }
 
  unsigned int sd= input.second-input.first; 
  unsigned int pd= p-v_detids.begin();

  unsigned int ibegin=v_gains.size();
  unsigned int iend  =v_gains.size()+sd;
  v_detids.insert(p,DetId);
  v_ibegin.insert(v_ibegin.begin()+pd,ibegin);
  v_iend.insert(v_iend.begin()+pd,iend);

  v_gains.insert(v_gains.end(),input.first,input.second);
  return true;
}

const SiStripApvGain::Range SiStripApvGain::getRange(const uint32_t DetId) const {
  // get SiStripApvGain Range of DetId
  RegistryConstIterator p = std::lower_bound(v_detids.begin(),v_detids.end(),DetId);
  if (p==v_detids.end() || *p!=DetId) 
    return SiStripApvGain::Range(v_gains.end(),v_gains.end()); 
  else{ 
    unsigned int pd= p-v_detids.begin();
    unsigned int ibegin = *(v_ibegin.begin()+pd);
    unsigned int iend   = *(v_iend.begin()+pd);
    __builtin_prefetch((&v_gains.front())+ibegin);
    return SiStripApvGain::Range(v_gains.begin()+ibegin,v_gains.begin()+iend);
  }
}

SiStripApvGain::Range  SiStripApvGain::getRangeByPos(unsigned short pos) const {
  if (pos>v_detids.size()) return Range(v_gains.end(),v_gains.end());
    unsigned int ibegin = *(v_ibegin.begin()+pos);
    unsigned int iend   = *(v_iend.begin()+pos);
    __builtin_prefetch((&v_gains.front())+ibegin);
    return SiStripApvGain::Range(v_gains.begin()+ibegin,v_gains.begin()+iend);
}


void SiStripApvGain::getDetIds(std::vector<uint32_t>& DetIds_) const {
  // returns vector of DetIds in map
  //  DetIds_=v_detids;
  DetIds_.insert(DetIds_.begin(),v_detids.begin(),v_detids.end());
}

#ifdef EDM_ML_DEBUG
float SiStripApvGain::getStripGain(const uint16_t& strip, const Range& range) {
  uint16_t apv = (uint16_t) (strip/128);
  if (apv>=range.second-range.first){
    throw cms::Exception("CorruptedData")
      << "[SiStripApvGain::getApvGain] looking for SiStripApvGain for a strip out of range: strip " << strip << " apv " << apv << std::endl;
  }
  
  //  return static_cast<float> (*(range.first+apv));

  return *(range.first+apv);

}

float SiStripApvGain::getApvGain(const uint16_t& apv, const Range& range) {
  if (apv>=range.second-range.first){
    throw cms::Exception("CorruptedData")
      << "[SiStripApvGain::getApvGain] looking for SiStripApvGain for an apv out of range: apv " << apv << std::endl;
  }
  
  //  return static_cast<float> (*(range.first+apv));

  return *(range.first+apv);
}
#endif


void SiStripApvGain::printDebug(std::stringstream & ss, const TrackerTopology* /*trackerTopo*/) const
{
  std::vector<unsigned int>::const_iterator detid = v_detids.begin();
  ss << "Number of detids " << v_detids.size() << std::endl;

  for( ; detid != v_detids.end(); ++detid ) {
    SiStripApvGain::Range range = getRange(*detid);
    int apv=0;
    for( int it=0; it < range.second - range.first; ++it ) {
      ss << "detid " << *detid << " \t"
         << " apv " << apv++ << " \t"
         << getApvGain(it,range) << " \t" 
         << std::endl;          
    } 
  }
}

void SiStripApvGain::printSummary(std::stringstream & ss, const TrackerTopology* trackerTopo) const
{
  SiStripDetSummary summaryGain{trackerTopo};

  std::vector<uint32_t>::const_iterator detid = v_detids.begin();
  for( ; detid != v_detids.end(); ++detid ) {
    Range range = getRange(*detid);
    for( int it=0; it < range.second - range.first; ++it ) {
      summaryGain.add(*detid, getApvGain(it, range));
    } 
  }
  ss << "Summary of gain values:" << std::endl;
  summaryGain.print(ss, true);
}
