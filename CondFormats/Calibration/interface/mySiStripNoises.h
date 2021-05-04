#ifndef mySiStripNoises_h
#define mySiStripNoises_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <map>
#include <cstdint>
//#include<iostream>

//typedef float SiStripNoise;
//typedef bool  SiStripDisable;

class mySiStripNoises {
public:
  mySiStripNoises() {}
  ~mySiStripNoises() {}

  struct DetRegistry {
    uint32_t detid;
    uint32_t ibegin;
    uint32_t iend;

    COND_SERIALIZABLE;
  };
  class StrictWeakOrdering {
  public:
    bool operator()(const DetRegistry& p, const uint32_t& i) const { return p.detid < i; }
  };
  typedef std::vector<unsigned char> SiStripNoiseVector;
  typedef SiStripNoiseVector::const_iterator ContainerIterator;
  typedef std::pair<ContainerIterator, ContainerIterator> Range;
  typedef std::vector<DetRegistry> Registry;
  typedef Registry::const_iterator RegistryIterator;
  typedef const std::vector<short> InputVector;

  bool put(const uint32_t detID, InputVector& input);
  const Range getRange(const uint32_t& detID) const;
  void getDetIds(std::vector<uint32_t>& DetIds) const;
  float getNoise(const uint16_t& strip, const Range& range) const;
  void setData(float noise_, std::vector<short>& vped);
  // private:
  //SiStripNoiseVector v_noises;
  //Registry indexes;
  void encode(InputVector& Vi, std::vector<unsigned char>& Vo_CHAR);
  uint16_t decode(const uint16_t& strip, const Range& range) const;
  std::vector<unsigned char> v_noises;
  std::vector<DetRegistry> indexes;

  COND_SERIALIZABLE;
};

#endif
