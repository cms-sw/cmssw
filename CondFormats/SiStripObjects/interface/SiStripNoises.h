#ifndef SiStripNoises_h
#define SiStripNoises_h

#include<vector>
#include<map>
#include<iostream>
#include<boost/cstdint.hpp>


class SiStripNoises {

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
  

  typedef std::vector<short>::const_iterator         ContainerIterator;  
  typedef std::pair<ContainerIterator, ContainerIterator>  Range;      
  typedef std::vector<DetRegistry>                         Registry;
  typedef Registry::const_iterator                         RegistryIterator;
 
  SiStripNoises(){};
  ~SiStripNoises(){};
    
  bool put(const uint32_t& detID,Range input);
  const Range getRange(const uint32_t& detID) const;
  void getDetIds(std::vector<uint32_t>& DetIds_) const;

  float   getNoise  (const uint16_t& strip, const Range& range) const;
  bool    getDisable(const uint16_t& strip, const Range& range) const;
  void    setData(float noise_, bool disable_, std::vector<short>& vped);
 
 private:
  std::vector<short> v_noises; //@@@ blob streaming doesn't work with uint16_t and with SiStripNoises::Data
  std::vector<DetRegistry> indexes;
};

#endif
