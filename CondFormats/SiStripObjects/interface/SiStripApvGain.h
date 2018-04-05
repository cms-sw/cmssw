#ifndef SiStripApvGain_h
#define SiStripApvGain_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include<vector>
#include<map>
#include<iostream>
#include<boost/cstdint.hpp>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class TrackerTopology;

/**
 * Stores the information of the gain for each apv using four vectors <br>
 * A vector<unsigned int> (v_detids) stores the detId. <br>
 * A vector<float> (v_gains) stores the value of the gain (more than one per detId). <br>
 * Two vector<unsigned int> (v_ibegin and v_iend) store the correspondence of the v_detids
 * and the ranges of values in v_gain. <br>
 *
 * The printSummary method uses SiStripDetSummary. See description therein. <br>
 * The printDebug method prints the gain value for every apv of every detId. <br>
 */

class SiStripApvGain {

 public:

  typedef std::vector<float>::const_iterator               ContainerIterator;  
  typedef std::pair<ContainerIterator, ContainerIterator>  Range;      
  typedef std::vector<unsigned int>                        Registry;
  typedef Registry::iterator                               RegistryIterator;
  typedef Registry::const_iterator                         RegistryConstIterator;
  typedef std::vector<float>                               InputVector;


  struct RegistryPointers{
    RegistryConstIterator detid_begin;
    RegistryConstIterator detid_end;
    RegistryConstIterator ibegin_begin;
    RegistryConstIterator ibegin_end;
    RegistryConstIterator iend_begin;
    RegistryConstIterator iend_end;
    ContainerIterator v_begin;
    ContainerIterator v_end;

    ContainerIterator getFirstElement(RegistryConstIterator& idet){return v_begin+*(ibegin_begin+(idet-detid_begin));} 
    ContainerIterator getLastElement(RegistryConstIterator& idet){return v_begin+*(iend_begin+(idet-detid_begin));} 
  };

  SiStripApvGain(){}
  ~SiStripApvGain(){}

  RegistryPointers getRegistryPointers() const {
    RegistryPointers p;
    p.detid_begin=v_detids.begin();
    p.detid_end=v_detids.end();
    p.ibegin_begin=v_ibegin.begin();
    p.ibegin_end=v_ibegin.end();
    p.iend_begin=v_iend.begin();
    p.iend_end=v_iend.end();
    p.v_begin=v_gains.begin();
    p.v_end=v_gains.end();

    return p;
}

  
  bool put(const uint32_t& detID, Range input);
  const Range getRange(const uint32_t  detID) const;
  Range getRangeByPos(unsigned short pos) const;
  void getDetIds(std::vector<uint32_t>& DetIds_) const;
  

#ifdef EDM_ML_DEBUG
  static float   getStripGain     (const uint16_t& strip, const Range& range);
  static float   getApvGain  (const uint16_t& apv, const Range& range);
#else
  static float   getStripGain (uint16_t strip, const Range& range)  {uint16_t apv = strip/128; return *(range.first+apv);}
  static float   getApvGain   (uint16_t apv, const Range& range) {return *(range.first+apv);}
#endif


  void printDebug(std::stringstream & ss, const TrackerTopology* trackerTopo) const;
  void printSummary(std::stringstream & ss, const TrackerTopology* trackerTopo) const;

 private:

  std::vector<float> v_gains;
  std::vector<unsigned int>   v_detids;
  std::vector<unsigned int>   v_ibegin;
  std::vector<unsigned int>   v_iend;

 COND_SERIALIZABLE;
};

#endif
