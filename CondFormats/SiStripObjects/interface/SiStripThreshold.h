#ifndef SiStripThreshold_h
#define SiStripThreshold_h

#include<vector>
#include<map>
#include<iostream>
#include<boost/cstdint.hpp>
#include "DataFormats/SiStripCommon/interface/ConstantsForCondObjects.h"


class SiStripThreshold {

 public:

  struct data{
    unsigned short firstStrip:10;
    unsigned short stripRange:10;
    unsigned int lowTh:6;
    unsigned int highTh :6;
  };

  struct DetRegistry{
    uint32_t detid;
    uint32_t ibegin;
    uint32_t iend;
  };




  class StrictWeakOrdering{
  public:
    bool operator() (const DetRegistry& p,const uint32_t& i) const {return p.detid < i;}
  };
  

  typedef std::vector<unsigned int>::const_iterator        ContainerIterator;  
  typedef std::pair<ContainerIterator, ContainerIterator>  Range;      
  typedef std::vector<DetRegistry>                         Registry;
  typedef Registry::const_iterator                         RegistryIterator;
 
  SiStripThreshold(){};
  virtual ~SiStripThreshold(){};
  
  bool put(const uint32_t& detID,Range input);
  const Range getRange(const uint32_t& detID) const;
  void getDetIds(std::vector<uint32_t>& DetIds_) const;
  
  
  ContainerIterator getDataVectorBegin()    const {return v_threshold.begin();}
  ContainerIterator getDataVectorEnd()      const {return v_threshold.end();}
  RegistryIterator getRegistryVectorBegin() const {return indexes.begin();}
  RegistryIterator getRegistryVectorEnd()   const{return indexes.end();}

  void  setData(float lth, float hth, std::vector<unsigned int>& vped);
  //float getLowTh (const uint16_t& strip, const Range& range) const;
  //float getHighTh(const uint16_t& strip, const Range& range) const;

  inline data decode (const unsigned int& value) const {
    data a;
    a.firstStrip = ((value>>sistrip::FirstThStripShift_)&sistrip::FirstThStripMask_);
    a.stripRange = ((value>>sistrip::RangeThStripShift_)&sistrip::RangeThStripMask_);
    a.lowTh      = ((value>>sistrip::LowThStripShift_)&sistrip::LowThStripMask_);
    a.highTh 	 = ((value>>sistrip::HighThStripShift_)&sistrip::HighThStripMask_);
    return a;
  }
  
  inline unsigned int encode (const unsigned int& strip, const unsigned short& NconsecutiveValueTh, const float& lTh,const float& hTh) {
    return   ((strip & sistrip::FirstThStripMask_)<<sistrip::FirstThStripShift_) | ((NconsecutiveValueTh & sistrip::RangeThStripMask_)<<sistrip::RangeThStripShift_) | (((uint32_t)lTh & sistrip::LowThStripMask_)<<sistrip::LowThStripShift_) | (((uint32_t)hTh & sistrip::HighThStripMask_)<<sistrip::HighThStripShift_);
  }

protected:
  std::vector<unsigned int> v_threshold; 
  std::vector<DetRegistry> indexes;
};

#endif
