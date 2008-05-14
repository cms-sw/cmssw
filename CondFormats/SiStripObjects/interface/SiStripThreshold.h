#ifndef SiStripThreshold_h
#define SiStripThreshold_h

#include<vector>
#include<map>
#include<iostream>
#include<boost/cstdint.hpp>
#include "DataFormats/SiStripCommon/interface/ConstantsForCondObjects.h"


class SiStripThreshold {

 public:

  struct Data{
    inline void encode (const uint16_t& strip, const float& lTh,const float& hTh) {        
      FirstStrip_and_Hth = 
	((strip & sistrip::FirstThStripMask_)<<sistrip::FirstThStripShift_) |
	((uint32_t)(hTh*5.0+0.5) & sistrip::HighThStripMask_);
      
      lowTh=((uint32_t)(lTh*5.0+0.5) & sistrip::LowThStripMask_);
    }
    
    inline uint16_t getFirstStrip() const {return (FirstStrip_and_Hth>>sistrip::FirstThStripShift_);}
    inline float  getHth() const {return (FirstStrip_and_Hth& sistrip::HighThStripMask_)/5.0;}
    inline float  getLth()const {return (lowTh& sistrip::LowThStripMask_)/5.0;}

    bool operator == (const Data& d) const { return (getHth() == d.getHth()) && (lowTh == d.lowTh) ; } 
    bool operator  < (const Data& d) const { return (FirstStrip_and_Hth  < d.FirstStrip_and_Hth) && (lowTh  < d.lowTh) ; } 

    uint16_t FirstStrip_and_Hth;
    uint8_t lowTh;
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

  class dataStrictWeakOrdering{
  public:
    bool operator() (const uint16_t& i, const Data& p) const {return i<p.FirstStrip_and_Hth ;}
  };
  

  typedef std::vector<Data>                                Container;  
  typedef Container::const_iterator                        ContainerIterator;  
  typedef std::pair<ContainerIterator, ContainerIterator>  Range;      
  typedef std::vector<DetRegistry>                         Registry;
  typedef Registry::const_iterator                         RegistryIterator;
  typedef Container                                        InputVector;  
 
  SiStripThreshold(){};
  SiStripThreshold(const SiStripThreshold& orig){
    v_threshold=orig.v_threshold; 
    indexes=orig.indexes;
  }
  virtual ~SiStripThreshold(){};
  
  bool put(const uint32_t& detID,InputVector vect);
  const Range getRange(const uint32_t& detID) const;
  void getDetIds(std::vector<uint32_t>& DetIds_) const;
  
  
  ContainerIterator getDataVectorBegin()     const {return v_threshold.begin();}
  ContainerIterator getDataVectorEnd()       const {return v_threshold.end();}
  RegistryIterator  getRegistryVectorBegin() const {return indexes.begin();}
  RegistryIterator  getRegistryVectorEnd()   const{return indexes.end();}

  void  setData(const uint16_t& strip, const float& lTh,const float& hTh, Container& vthr);
  SiStripThreshold::Data getData (const uint16_t& strip, const Range& range) const;
  
 private:
  
  Container::iterator compact(Container& input);

 private:
  Container v_threshold; 
  Registry indexes;
};

#endif
