#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

bool SiStripApvGain::put(const uint32_t& DetId, Range input) {
  // put in SiStripApvGain of DetId

  RegistryIterator p = std::lower_bound(v_detids.begin(),v_detids.end(),DetId);
  if (p!=v_detids.end() && *p==(int)DetId){
    edm::LogError("SiStripApvGain") << "[" << __PRETTY_FUNCTION__ << "] SiStripApvGain for DetID " << DetId << " is already stored. Skippig this put" << std::endl;
    return false;
  }
 
  int sd= input.second-input.first; 
  int pd= p-v_detids.begin();

  int ibegin=v_gains.size();
  int iend  =v_gains.size()+sd;
  v_detids.insert(p,DetId);
  v_ibegin.insert(v_ibegin.begin()+pd,ibegin);
  v_iend.insert(v_iend.begin()+pd,iend);

  v_gains.insert(v_gains.end(),input.first,input.second);
  return true;
}

const SiStripApvGain::Range SiStripApvGain::getRange(const uint32_t& DetId) const {
  // get SiStripApvGain Range of DetId
  
  RegistryConstIterator p = std::lower_bound(v_detids.begin(),v_detids.end(),DetId);
  if (p==v_detids.end() || *p!=(int)DetId) 
    return SiStripApvGain::Range(v_gains.end(),v_gains.end()); 
  else{ 
    int pd= p-v_detids.begin();
    int ibegin = *(v_ibegin.begin()+pd);
    int iend   = *(v_iend.begin()+pd);    
    return SiStripApvGain::Range(v_gains.begin()+ibegin,v_gains.begin()+iend);
  }
}

void SiStripApvGain::getDetIds(std::vector<uint32_t>& DetIds_) const {
  // returns vector of DetIds in map
  //  DetIds_=v_detids;
  DetIds_.insert(DetIds_.begin(),v_detids.begin(),v_detids.end());
}

float SiStripApvGain::getStripGain(const uint16_t& strip, const Range& range) const {
  uint16_t apv = (uint16_t) (strip/128);
  if (apv>=range.second-range.first){
    throw cms::Exception("CorruptedData")
      << "[SiStripApvGain::getApvGain] looking for SiStripApvGain for a strip out of range: strip " << strip << " apv " << apv << std::endl;
  }
  
  return static_cast<float> (*(range.first+apv));
}

float SiStripApvGain::getApvGain(const uint16_t& apv, const Range& range) const {
  if (apv>=range.second-range.first){
    throw cms::Exception("CorruptedData")
      << "[SiStripApvGain::getApvGain] looking for SiStripApvGain for an apv out of range: apv " << apv << std::endl;
  }
  
  return static_cast<float> (*(range.first+apv));
}

