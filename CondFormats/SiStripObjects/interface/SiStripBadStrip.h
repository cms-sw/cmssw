#ifndef SiStripBadStrip_h
#define SiStripBadStrip_h

#include<vector>
#include<map>
#include<iostream>
#include<boost/cstdint.hpp>


class SiStripBadStrip {

 public:

  struct DetRegistry{
    uint32_t detid;
    uint32_t ibegin;
    uint32_t iend;
  };

  class StrictWeakOrdering{
  public:
    bool operator() (const DetRegistry& p,const uint32_t& i) const {return p.detid < i;}
  };
  

  typedef std::vector<int>::const_iterator        ContainerIterator;  
  typedef std::pair<ContainerIterator, ContainerIterator>  Range;      
  typedef std::vector<DetRegistry>                         Registry;
  typedef Registry::const_iterator                         RegistryIterator;
 
  SiStripBadStrip(){};
  ~SiStripBadStrip(){};
  
  bool put(const uint32_t& detID,Range input);
  const Range getRange(const uint32_t& detID) const;
  void getDetIds(std::vector<uint32_t>& DetIds_) const;

  int  getBadStrips(const Range& range) const;
  //void getBadApvs(const Range& range, std::vector<short>& );

 private:
  std::vector<int> v_badstrips; 
  std::vector<DetRegistry> indexes;
};

#endif
