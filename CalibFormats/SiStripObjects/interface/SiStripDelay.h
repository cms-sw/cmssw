#ifndef SiStripObjects_SiStripDelay_h
#define SiStripObjects_SiStripDelay_h
// -*- C++ -*-
//
// Package:     SiStripObjects
// Class  :     SiStripDelay
// 
/**
 * Author: M. De Mattia (demattia@pd.infn.it) 25/10/2010:
 *
 * Dependent record used to combine the SiStripBaseDelays and provide the reconstruction
 * with a single delay value. <br>
 * When the object is built the list of values is stored in boost::unordered_map which,
 * for the number of modules in the Tracker (~15000) resulted much faster (more than a factor 2)
 * than a std::map or a lower_bound search in a sorted vector. <br>
 * The base delays must be filled in together with the sign they will get in the summation:
 * baseDelay1*sign1 + baseDelay2*sign2 + ... <br>
 * Pointers to the baseDelays are stored and after the fill is complete the method "makeDelay"
 * must be called to build the internal map. <br>
 * This additional step is required such that we don't build the map anytime a new baseDelay is inserted
 * and we don't make checks anytime the getDelay method is called. <br>
 * NOTE: Even if the code does not rely on the presence of the same detIds in all the baseDelays, this
 * condition should be fullfilled for consistency. The code checks only that the number of detIds
 * is the same in all baseDelays.
 */

#include "CondFormats/SiStripObjects/interface/SiStripBaseDelay.h"
#include <vector>
#include <memory>
#include <boost/unordered_map.hpp>

class SiStripDelay
{
 public:
  SiStripDelay() {};
  virtual ~SiStripDelay() {};

  SiStripDelay(const SiStripDelay&) = delete;
  const SiStripDelay& operator=(const SiStripDelay&) = delete;

  inline SiStripDelay(const SiStripBaseDelay& baseDelay, const int sumSign,
		      const std::pair<std::string, std::string> & recordLabelPair)
  {
    fillNewDelay(baseDelay, sumSign, recordLabelPair);
  }

  void fillNewDelay(const SiStripBaseDelay& baseDelay, const int sumSign,
		    const std::pair<std::string, std::string> & recordLabelPair);

  /// Return the delay combining all the baseDelays
  float getDelay(const uint32_t detId) const;

  /// Builds the boost::unordered_map
  bool makeDelay();

  /// Empty all the containers
  void clear();
  
  /**
   * The second parameter allows to specify which delay to retrieve, considering that
   * they are in input order.
   * NOTE that no protection is inside the method (because we want to keep it very light)
   * therefore it is the caller duty to check that the index is in the correct range.
   */
  inline const SiStripBaseDelay * getBaseDelay(const uint32_t index) const
  {
    return baseDelayVector_[index];
  }

  inline size_t getNumberOfTags() const
  {
    return baseDelayVector_.size();
  }
  inline std::string getRcdName(const uint32_t index) const
  {
    return recordLabelPair_[index].first;
  }
  inline std::string getLabelName(const uint32_t index) const
  {
    return recordLabelPair_[index].second;
  }
  inline int getTagSign(const uint32_t index) const
  {
    return sumSignVector_[index];
  }

  /// Prints the average value of the delays for all layers and wheels in the SiStripTracker
  void printSummary(std::stringstream& ss, const TrackerTopology* trackerTopo) const;
  /// Prints the delays for all the detIds
  void printDebug(std::stringstream& ss, const TrackerTopology* tTopo) const;

 private:
  // ---------- member data --------------------------------

  std::vector<const SiStripBaseDelay *> baseDelayVector_;
  std::vector<int> sumSignVector_;
  std::vector<std::pair<std::string, std::string> > recordLabelPair_;
  boost::unordered_map<uint32_t, double> delays_;
};

#endif
