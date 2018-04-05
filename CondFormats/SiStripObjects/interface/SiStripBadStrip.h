#ifndef SiStripBadStrip_h
#define SiStripBadStrip_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include<vector>
#include<map>
#include<iostream>
#include<boost/cstdint.hpp>
#include "DataFormats/SiStripCommon/interface/ConstantsForCondObjects.h"

class TrackerTopology;

/**
 * Holds the list of bad components. <br>
 * The bad components can be filled with two put methods, that receive a DetId and
 * either a range of bad strips or a vector of bad strips. <br>
 * The information is stored in two vectors: <br>
 * - a vector<unsigned int> holding the bad strips <br>
 * - a vector<DetRegistry> associating a DetId to the corresponding range of
 * bad strips in the vector described above <br>
 * A DetRegistry contains the DetId and the index of the first and last position
 * in the bad strips vector correspoding to this DetId. <br>
 * The printSummary method prints the number of modules with at least a bad strip
 * divided by subdetector and layer/disk. It also prints the number of bad strips
 * divided in the same way. <br>
 * The printDebug method prints the full information of all the bad strips.
 */


class SiStripBadStrip {

 public:

  struct data{
    unsigned short firstStrip;
    unsigned short range;
    unsigned short flag;
  };

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

  typedef std::vector<unsigned int>                        Container;
  typedef std::vector<unsigned int>::const_iterator        ContainerIterator;
  typedef std::pair<ContainerIterator, ContainerIterator>  Range;
  typedef std::vector<DetRegistry>                         Registry;
  typedef Registry::const_iterator                         RegistryIterator;
  typedef Container                                        InputVector;
 
  SiStripBadStrip(){};
  SiStripBadStrip(const SiStripBadStrip& orig){
    v_badstrips=orig.v_badstrips; 
    indexes=orig.indexes;
  }
  virtual ~SiStripBadStrip(){};

  bool put(const uint32_t& detID,const InputVector& vect){return put(detID,Range(vect.begin(),vect.end()));}
  bool put(const uint32_t& detID,Range input);
  const Range getRange(const uint32_t detID) const;
  Range getRangeByPos(unsigned short pos) const;
  void getDetIds(std::vector<uint32_t>& DetIds_) const;
  void printSummary(std::stringstream & ss, const TrackerTopology* trackerTopo) const;
  void printDebug(std::stringstream & ss, const TrackerTopology* trackerTopo) const;

  ContainerIterator getDataVectorBegin()    const {return v_badstrips.begin();}
  ContainerIterator getDataVectorEnd()      const {return v_badstrips.end();}
  RegistryIterator getRegistryVectorBegin() const {return indexes.begin();}
  RegistryIterator getRegistryVectorEnd()   const{return indexes.end();}


  inline data decode (const unsigned int& value) const {
    data a;
    a.firstStrip = ((value>>sistrip::FirstBadStripShift_)&sistrip::FirstBadStripMask_);
    a.range      = ((value>>sistrip::RangeBadStripShift_)&sistrip::RangeBadStripMask_);
    a.flag       = ((value>>sistrip::FlagBadStripShift_)&sistrip::FlagBadStripMask_);
    return a;
  }
  
  inline unsigned int encode (const unsigned short& first, const unsigned short& NconsecutiveBadStrips, const unsigned short& flag=0) {
    return   ((first & sistrip::FirstBadStripMask_)<<sistrip::FirstBadStripShift_) | ((NconsecutiveBadStrips & sistrip::RangeBadStripMask_)<<sistrip::RangeBadStripShift_) | ((flag & sistrip::FlagBadStripMask_)<<sistrip::FlagBadStripShift_);
  }

protected:
  Container v_badstrips; 
  Registry indexes;

 COND_SERIALIZABLE;
};

#endif
