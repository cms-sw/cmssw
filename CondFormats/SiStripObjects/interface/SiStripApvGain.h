#ifndef SiStripApvGain_h
#define SiStripApvGain_h

#include<vector>
#include<map>
#include<iostream>
#include<boost/cstdint.hpp>


class SiStripApvGain {

	public:

  typedef std::vector<float>::const_iterator               ContainerIterator;  
  typedef std::pair<ContainerIterator, ContainerIterator>  Range;      
  typedef std::vector<unsigned int>                        Registry;
  typedef Registry::iterator                               RegistryIterator;
  typedef Registry::const_iterator                         RegistryConstIterator;
 
  SiStripApvGain(){};
  ~SiStripApvGain(){};
  
  bool put(const uint32_t& detID, Range input);
  const Range getRange(const uint32_t& detID) const;
  void getDetIds(std::vector<uint32_t>& DetIds_) const;

  float   getStripGain     (const uint16_t& strip, const Range& range) const;
  float   getApvGain  (const uint16_t& apv, const Range& range) const;

 private:

  std::vector<float> v_gains; 
  std::vector<unsigned int>   v_detids;
  std::vector<unsigned int>   v_ibegin;
  std::vector<unsigned int>   v_iend;
};

#endif
