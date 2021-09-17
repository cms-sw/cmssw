#ifndef SiStripThreshold_h
#define SiStripThreshold_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <map>
#include <iostream>
#include "DataFormats/SiStripCommon/interface/ConstantsForCondObjects.h"
#include <sstream>
#include <cstdint>

class TrackerTopology;

/**
 * Holds the thresholds:<br>
 * - High threshold <br>
 * - Low threshold <br>
 * - Cluster threshold <br>
 * The values are stored as bitsets, in particular:
 * - One uint16_t stores the first strip and the high threshold.
 * - One uint8_t stores the low threshold.
 * - One uint8_t stores the cluster threshold.
 * The information are stored in a Data struct, which is also responsible
 * for the encoding and decoding. <br>
 * To fill the SiStripThreshold object:
 * - create and empty vector<Data>: SiStripThreshold::Container theSiStripVector;   
 * - use the setData method to fill it with the thresholds
 * - use the put method to save the values in the object.
 * The put method is used to actually fill the object. It receives the DetId and the vector<Data>
 * with the threshold values.<br>
 * Before being saved the vector<Data> is sorted in the FirstStrip and consecutive entries having
 * all the same thresholds are removed. This way it still stores the same information using less space.<br>
 * To retrieve the information:
 * - getDetIds: fills a vector with all detIds for which the thresholds have been stored.
 * - getDataVectorBegin and getDataVectorEnd: return the begin and end iterators to the vector<Data>.
 * - getData can be used to get the thresholds for a single strip.
 * The printSummary method prints mean, rms, min and max threshold values for each DetId.
 * The printDebug method prints all the thresholds for all DetIds.
 */

class SiStripThreshold {
public:
  struct Data {
    //used to create the threshold object for the ZS (that has only 2 thresholds)
    inline void encode(const uint16_t& strip, const float& lTh, const float& hTh) {
      FirstStrip_and_Hth = ((strip & sistrip::FirstThStripMask_) << sistrip::FirstThStripShift_) |
                           ((uint32_t)(hTh * 5.0 + 0.5) & sistrip::HighThStripMask_);

      lowTh = ((uint32_t)(lTh * 5.0 + 0.5) & sistrip::LowThStripMask_);
      clusTh = 0;  //put as default;
    }

    inline void encode(const uint16_t& strip, const float& lTh, const float& hTh, const float& cTh) {
      encode(strip, lTh, hTh);
      clusTh = (uint8_t)(cTh * 10 + .5);
    }

    inline uint16_t getFirstStrip() const { return (FirstStrip_and_Hth >> sistrip::FirstThStripShift_); }
    inline float getHth() const { return (FirstStrip_and_Hth & sistrip::HighThStripMask_) / 5.0; }
    inline float getLth() const { return (lowTh & sistrip::LowThStripMask_) / 5.0; }
    inline float getClusth() const { return clusTh / 10.0; }

    bool operator==(const Data& d) const {
      return (getHth() == d.getHth()) && (lowTh == d.lowTh) && (clusTh == d.clusTh);
    }
    bool operator<(const Data& d) const { return (FirstStrip_and_Hth < d.FirstStrip_and_Hth); }

    void print(std::stringstream& ss) const {
      ss << "firstStrip: " << getFirstStrip() << " \t"
         << "lTh: "
         << " " << getLth() << " \t"
         << "hTh: "
         << " " << getHth() << " \t"
         << "cTh: "
         << " " << getClusth() << " \t";
    }

    uint16_t FirstStrip_and_Hth;
    uint8_t lowTh;
    uint8_t clusTh;

    COND_SERIALIZABLE;
  };

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

  class dataStrictWeakOrdering {
  public:
    bool operator()(const uint16_t& i, const Data& p) const { return i < p.FirstStrip_and_Hth; }
  };

  typedef std::vector<Data> Container;
  typedef Container::const_iterator ContainerIterator;
  typedef std::pair<ContainerIterator, ContainerIterator> Range;
  typedef std::vector<DetRegistry> Registry;
  typedef Registry::const_iterator RegistryIterator;
  typedef Container InputVector;

  SiStripThreshold(){};
  SiStripThreshold(const SiStripThreshold& orig) {
    v_threshold = orig.v_threshold;
    indexes = orig.indexes;
  }
  virtual ~SiStripThreshold(){};

  bool put(const uint32_t& detID, const InputVector& vect);
  const Range getRange(const uint32_t& detID) const;
  void getDetIds(std::vector<uint32_t>& DetIds_) const;

  ContainerIterator getDataVectorBegin() const { return v_threshold.begin(); }
  ContainerIterator getDataVectorEnd() const { return v_threshold.end(); }
  RegistryIterator getRegistryVectorBegin() const { return indexes.begin(); }
  RegistryIterator getRegistryVectorEnd() const { return indexes.end(); }

  void setData(const uint16_t& strip, const float& lTh, const float& hTh, Container& vthr);
  void setData(const uint16_t& strip, const float& lTh, const float& hTh, const float& cTh, Container& vthr);
  SiStripThreshold::Data getData(const uint16_t& strip, const Range& range) const;

  void allThresholds(std::vector<float>& lowThs, std::vector<float>& highThs, const Range& range) const;

  /// Prints mean, rms, min and max threshold values for each DetId.
  void printSummary(std::stringstream& ss, const TrackerTopology* trackerTopo) const;
  /// Prints all the thresholds for all DetIds.
  void printDebug(std::stringstream& ss, const TrackerTopology* trackerTopo) const;

private:
  Container::iterator compact(Container& input);
  void addToStat(float value, uint16_t& range, float& sum, float& sum2, float& min, float& max) const;

private:
  Container v_threshold;
  Registry indexes;

  COND_SERIALIZABLE;
};

#endif
