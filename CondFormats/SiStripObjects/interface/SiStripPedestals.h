#ifndef SiStripPedestals_h
#define SiStripPedestals_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include<vector>
#include<map>
#include<iostream>
#include<boost/cstdint.hpp>

// #include "CondFormats/SiStripObjects/interface/SiStripBaseObject.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"

/**
 * Stores the pedestal of every strip. <br>
 * Encodes the information in a vector<char> and uses a vector<DetRegistry> to
 * connect each range of values to the corresponding detId. <br>
 * The DetRegistry struct contains the detId and two uint32_t giving the index
 * of begin and end of the corresponding range in the vector<char>. <br>
 * Has methods to return the pedestal of a given strip of of all the strips. <br>
 *
 * The printSummary method uses SiStripDetSummary. See description therein. <br>
 * The printDebug method prints the pedestal value for every strip of every detId. <br>
 */

// class SiStripPedestals : public SiStripBaseObject
class SiStripPedestals
{
public:
  /*
    struct DecodingStructure{  
    unsigned int lth :6;
    unsigned int hth :6;
    unsigned int ped :10;
    };*/

  struct DetRegistry{
    uint32_t detid;
    uint32_t ibegin;
    uint32_t iend;
  
  COND_SERIALIZABLE;
};

  class StrictWeakOrdering{
  public:
    bool operator() (const DetRegistry& p,const uint32_t& i) const {return p.detid < i;}
  };

  typedef std::vector<char>                                Container;  
  typedef std::vector<char>::const_iterator                ContainerIterator;  
  typedef std::pair<ContainerIterator, ContainerIterator>  Range;      
  typedef std::vector<DetRegistry>                         Registry;
  typedef Registry::const_iterator                         RegistryIterator;
  typedef std::vector<uint16_t>	            	         InputVector;

  SiStripPedestals(){};
  ~SiStripPedestals(){};

  //bool  put(const uint32_t& detID,Range input);
  bool  put(const uint32_t& detID,InputVector &input);
  const Range getRange(const uint32_t& detID) const;
  void  getDetIds(std::vector<uint32_t>& DetIds_) const;

  ContainerIterator getDataVectorBegin()    const {return v_pedestals.begin();}
  ContainerIterator getDataVectorEnd()      const {return v_pedestals.end();}
  RegistryIterator getRegistryVectorBegin() const {return indexes.begin();}
  RegistryIterator getRegistryVectorEnd()   const{return indexes.end();}

  void  setData(float ped, InputVector& vped);
  float getPed   (const uint16_t& strip, const Range& range) const;
  void  allPeds  (std::vector<int> & pefs,  const Range& range) const;

  /// Prints mean pedestal value divided for subdet, layer and mono/stereo.
  void printSummary(std::stringstream& ss, const TrackerTopology* trackerTopo) const;
  /// Prints all pedestals.
  void printDebug(std::stringstream& ss, const TrackerTopology* trackerTopo) const;

 private:

  void     encode(InputVector& Vi, std::vector<unsigned char>& Vo_CHAR);
  uint16_t decode (const uint16_t& strip, const Range& range) const;
  inline uint16_t get10bits(const uint8_t * &ptr, int8_t skip) const ;

  Container v_pedestals; //@@@ blob streaming doesn't work with uint16_t and with SiStripData::Data
  Registry indexes;

 COND_SERIALIZABLE;
};

#endif
