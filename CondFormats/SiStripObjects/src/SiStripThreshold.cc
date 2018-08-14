#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <cassert>
#include <algorithm>
#include <cmath>

bool SiStripThreshold::put(const uint32_t& DetId, const InputVector& _vect) {
  InputVector vect = _vect;
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

void SiStripThreshold::setData(const uint16_t& strip, const float& lTh,const float& hTh, const float& cTh, Container& vthr){
  Data a;
  a.encode(strip,lTh,hTh,cTh);
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

void SiStripThreshold::allThresholds(std::vector<float> &lowThs, std::vector<float> &highThs, const Range& range) const  {
    ContainerIterator it = range.first;
    size_t strips = lowThs.size(); 
    assert(strips == highThs.size());
    while (it != range.second) {
        size_t firstStrip = it->getFirstStrip();
        //std::cout << "First strip is " << firstStrip << std::endl;
        float high = it->getHth(), low = it->getLth();
        //std::cout << "High is " << high << ", low is " << low << std::endl;
        ++it; // increment the pointer
        size_t lastStrip = (it == range.second ? strips : it->getFirstStrip());
        //std::cout << "Last strip is " << lastStrip << std::endl;
        if (lastStrip > strips) { 
            it = range.second;  // I should stop here,
            lastStrip = strips; // and fill only 'strips' strips
        }
        std::fill( & lowThs[firstStrip] , & lowThs[lastStrip] , low );
        std::fill( & highThs[firstStrip], & highThs[lastStrip], high );
    }
}    

void SiStripThreshold::printDebug(std::stringstream& ss, const TrackerTopology* /*trackerTopo*/) const{
  RegistryIterator rit=getRegistryVectorBegin(), erit=getRegistryVectorEnd();
  ContainerIterator it,eit;
  for(;rit!=erit;++rit){
    it=getDataVectorBegin()+rit->ibegin;
    eit=getDataVectorBegin()+rit->iend;
    ss << "\ndetid: " << rit->detid << " \t ";
    for(;it!=eit;++it){
      ss << "\n \t ";
      it->print(ss);
    }
  }
}

void SiStripThreshold::printSummary(std::stringstream& ss, const TrackerTopology* /*trackerTopo*/) const{
  RegistryIterator rit=getRegistryVectorBegin(), erit=getRegistryVectorEnd();
  ContainerIterator it,eit,itp;
  float meanLth, meanHth, meanCth; //mean value 
  float rmsLth, rmsHth, rmsCth; //rms value 
  float maxLth, maxHth, maxCth; //max value 
  float minLth, minHth, minCth; //min value 
  uint16_t n;
  uint16_t firstStrip,stripRange;
  for(;rit!=erit;++rit){
    it=getDataVectorBegin()+rit->ibegin;
    eit=getDataVectorBegin()+rit->iend;
    ss << "\ndetid: " << rit->detid << " \t ";

    meanLth=0; meanHth=0; meanCth=0; //mean value 
    rmsLth=0; rmsHth=0; rmsCth=0; //rms value 
    maxLth=0; maxHth=0; maxCth=0; //max value 
    minLth=10000; minHth=10000; minCth=10000; //min value 
    n=0;
    firstStrip=0;
    for(;it!=eit;++it){
      itp=it+1;
      firstStrip=it->getFirstStrip();
      if(itp!=eit)
	stripRange=(itp->getFirstStrip()-firstStrip);
      else 
	stripRange=firstStrip>511?768-firstStrip:512-firstStrip; //*FIXME, I dont' know ithis class the strip number of a detector, so I assume wrongly that if the last firstStrip<511 the detector has only 512 strips. Clearly wrong. to be fixed

      addToStat(it->getLth()   ,stripRange,meanLth,rmsLth,minLth,maxLth);
      addToStat(it->getHth()   ,stripRange,meanHth,rmsHth,minHth,maxHth);
      addToStat(it->getClusth(),stripRange,meanCth,rmsCth,minCth,maxCth);
      n+=stripRange;
    }
    meanLth/=n;
    meanHth/=n;
    meanCth/=n;
    rmsLth= sqrt(rmsLth/n-meanLth*meanLth);
    rmsHth= sqrt(rmsHth/n-meanHth*meanHth);
    rmsCth= sqrt(rmsCth/n-meanCth*meanCth);
    ss<< "\nn " << n << " \tmeanLth " << meanLth << " \t rmsLth " << rmsLth << " \t minLth " << minLth << " \t maxLth " << maxLth;  
    ss<< "\n\tmeanHth " << meanHth << " \t rmsHth " << rmsHth << " \t minHth " << minHth << " \t maxHth " << maxHth;  
    ss<< "\n\tmeanCth " << meanCth << " \t rmsCth " << rmsCth << " \t minCth " << minCth << " \t maxCth " << maxCth;  
  }
}

void SiStripThreshold::addToStat(float value, uint16_t& range, float& sum, float& sum2, float& min, float& max) const{
  sum+=value*range;
  sum2+=value*value*range;
  if(value<min)
    min=value;
  if(value>max)
    max=value;
}
