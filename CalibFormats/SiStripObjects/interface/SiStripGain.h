#ifndef SiStripObjects_SiStripGain_h
#define SiStripObjects_SiStripGain_h
// -*- C++ -*-
//
// Package:     SiStripObjects
// Class  :     SiStripGain
// 
/**\class SiStripGain SiStripGain.h CalibFormats/SiStripObjects/interface/SiStripGain.h
 *
 * Description: give detector view for the cabling classes
 *
 * Usage:
 *  <usage>
 *
 * Original Author:  gbruno
 *         Created:  Wed Mar 22 12:24:20 CET 2006
 *
 * Modifications by M. De Mattia (demattia@pd.infn.it) on 11/11/2009:
 * It now holds a std::vector of pointers to ApvGain and a std::vector of corresponding
 * normalization factors. <br>
 * It returns the product of all the Gain/norm ratios. <br>
 * The multiply method allows to input additional gain records. <br>
 * ATTENTION: the code assumes that the second tag has at least the same DetIds that the first tag and
 * only the DetIds present in the first tag will be used. <br>
 * <br>
 * There are two set of methods to access the gain value. The first one returns the products of all ApvGain/norm.
 * The second set of methods take an additional integer paramter and return the corresponding ApvGain (without normalization).
 * Note that no check is done inside these methods to see if the ApvGain really exists. It is responsibility of the
 * user to not pass an index value that exceeds the number of ApvGains. <br>
 * The normalization factors for each of the stored ApvGains are also accessible passing the corresponding index.
 * <br>
 * Additional method are provided to get the number of ApvGains used to build this object, the names of the records
 * that stored those ApvGains and the labels (they can be used to go back to the tags looking in the cfg).
 */

#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include <vector>
#include <memory>

class TrackerTopology;

class SiStripGain
{
 public:
  SiStripGain() {}
  SiStripGain(const SiStripGain&) = delete;
  const SiStripGain& operator=(const SiStripGain&) = delete;

  /// Kept for compatibility
  inline SiStripGain(const SiStripApvGain& apvgain, const double & factor) :
    apvgain_(nullptr)
  {
    multiply(apvgain, factor, std::make_pair("", ""));
  }

  inline SiStripGain(const SiStripApvGain& apvgain, const double & factor,
		     const std::pair<std::string, std::string> & recordLabelPair) :
    apvgain_(nullptr)
  {
    multiply(apvgain, factor, recordLabelPair);
  }

  /// Used to input additional gain values that will be multiplied to the first one
  void multiply(const SiStripApvGain & apvgain, const double & factor,
		const std::pair<std::string, std::string> & recordLabelPair);

  // getters
  // For the product of all apvGains
  // -------------------------------
  const SiStripApvGain::Range getRange(uint32_t detID) const { return apvgain_->getRange(detID);}
  SiStripApvGain::Range getRangeByPos(unsigned short pos) const { return apvgain_->getRangeByPos(pos);}
  static float getStripGain(const uint16_t& strip, const SiStripApvGain::Range& range) { return  SiStripApvGain::getStripGain(strip, range);}
  static float getApvGain(const uint16_t& apv, const SiStripApvGain::Range& range) {  return SiStripApvGain::getApvGain(apv, range); }


  // For a specific apvGain
  // ----------------------
  /**
   * The second parameter allows to specify which gain to retrieve, considering that
   * they are in input order.
   * NOTE that no protection is inside the method (because we want to keep it very light)
   * therefore it is the caller duty to check that the index is in the correct range.
   */
  const SiStripApvGain::Range getRange(const uint32_t& detID, const uint32_t index) const;
  float getStripGain(const uint16_t& strip, const SiStripApvGain::Range& range, const uint32_t index) const;
  float getApvGain(const uint16_t& apv, const SiStripApvGain::Range& range, const uint32_t index) const;

  /// ATTENTION: we assume the detIds are the same as those from the first gain
  void getDetIds(std::vector<uint32_t>& DetIds_) const;

  inline size_t getNumberOfTags() const
  {
    return apvgainVector_.size();
  }
  inline std::string getRcdName(const uint32_t index) const
  {
    return recordLabelPair_[index].first;
  }
  inline std::string getLabelName(const uint32_t index) const
  {
    return recordLabelPair_[index].second;
  }
  inline double getTagNorm(const uint32_t index) const
  {
    return normVector_[index];
  }

  void printDebug(std::stringstream& ss, const TrackerTopology* trackerTopo) const;
  void printSummary(std::stringstream& ss, const TrackerTopology* trackerTopo) const;

 private:

  void fillNewGain(const SiStripApvGain * apvgain, const double & factor,
		   const SiStripApvGain * apvgain2 = nullptr, const double & factor2 = 1.);

  // ---------- member data --------------------------------

  std::vector<const SiStripApvGain *> apvgainVector_;
  std::vector<double> normVector_;
  const SiStripApvGain * apvgain_;
  std::unique_ptr<SiStripApvGain> apvgainAutoPtr_;
  std::vector<std::pair<std::string, std::string> > recordLabelPair_;
};

#endif
