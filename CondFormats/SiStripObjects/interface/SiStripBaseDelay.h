#ifndef SiStripBaseDelay_h
#define SiStripBaseDelay_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <algorithm>
#include <cstdint>
#include <sstream>
#include <boost/bind/bind.hpp>

#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"

/**
 * Author: M. De Mattia (demattia@pd.infn.it) 25/10/2010:
 *
 * Base Delay object containing the PLL or time of flight delays. <br>
 * It stores the values in a vector\<Delay\>, which is not sorted. <br>
 * This object can be used directly to access the information on the delays for each
 * detId. However, it is recommended to use the SiStripDelay dependent record which is
 * able to combine multiple BaseDelays and provides a much faster access to the information.
 */

class SiStripBaseDelay
{
 public:

  SiStripBaseDelay() {}

  // Defined as public for genreflex
  struct Delay
  {
    Delay(const uint32_t inputDetId, const uint16_t inputCoarseDelay, const uint16_t inputFineDelay) :
      detId(inputDetId),
      coarseDelay(inputCoarseDelay),
      fineDelay(inputFineDelay)
    {}
    /// Default constructor needed by genreflex
    Delay() :
      detId(0),
      coarseDelay(255),
      fineDelay(255)
    {}
    uint32_t detId;
    unsigned char coarseDelay;
    unsigned char fineDelay;
  
  COND_SERIALIZABLE;
};
  typedef std::vector<Delay>::iterator delayIt;
  typedef std::vector<Delay>::const_iterator delayConstIt;

  bool put( const uint32_t detId, const uint16_t coarseDelay, const uint16_t fineDelay );
  uint16_t coarseDelay(const uint32_t detId);
  uint16_t fineDelay(const uint32_t detId) const;
  double delay(const uint32_t detId) const;

  /// Fill the input container with all the delays
  void delays(std::vector<Delay> & delays) const
  {
    delays = delays_;
  }

  /// Get the list of all detIds for which a delay is stored
  void detIds(std::vector<uint32_t> & detIdVector) const;

  /// Get the total number of delays stored (should equal the total number of modules in the SiStripTracker)
  inline uint32_t delaysSize() const
  {
    return delays_.size();
  }

  /// Prints the average value of the delays for all layers and wheels in the SiStripTracker
  void printSummary(std::stringstream & ss, const TrackerTopology* trackerTopo) const;
  /// Prints the delays for all the detIds
  void printDebug(std::stringstream & ss, const TrackerTopology* trackerTopo) const;

 private:

  inline double makeDelay(const uint16_t coarseDelay, const uint16_t fineDelay) const
  {
    return( coarseDelay*25 + fineDelay*(25/24.) );
  }

  std::vector<Delay> delays_;

 COND_SERIALIZABLE;
};

#endif
