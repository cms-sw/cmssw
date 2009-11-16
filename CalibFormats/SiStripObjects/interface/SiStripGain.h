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
 * $Id: SiStripGain.h,v 1.4 2009/11/13 16:19:36 demattia Exp $
 *
 * Modifications by M. De Mattia (demattia@pd.infn.it) on 11/11/2009:
 * It now holds a vector of pointers to ApvGain and a vector of corresponding
 * normalization factors. <br>
 * It returns the product of all the Gain/norm ratios. <br>
 * The multiply method allows to input additional gain records. <br>
 * ATTENTION: we take the list of detIds from the first gain and we assume
 * that is it the same for all the rest of the gains. <br>
 * <br>
 * The getStripGain and getApvGain return the gain value for the selected range of a gain set
 * specified by index in the internal vector (default 0). This vector is built adding each
 * gain in sequence and retains this order, so the first set of gain input will be 0 ecc... <br>
 * Additional overloaded methods receive the full vector of ranges for all the gains and returns
 * the product of all gains/norm. <br>
 * The getRange method can be used to take the range for a given detId and index of the gain set
 * (default 0). <br>
 * The getAllRanges method can be used to take all the ranges for a given detId and pass them to
 * the getStripGain and getApvGain methods.
 * The ESProducer CalibTracker/SiStripESProducers/plugins/real/SiStripGainESProducerTemplate.h
 * handles the multiple inputs of gains and shows an example on how to use this class.
 */

#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include <vector>

using namespace std;

class SiStripGain
{
 public:
  SiStripGain() {};
  virtual ~SiStripGain() {};

  inline SiStripGain(const SiStripApvGain& apvgain, const double & factor)
  {
    multiply(apvgain, factor);
  }

  /// Used to input additional gain values that will be multiplied to the first one
  void multiply(const SiStripApvGain & apvgain, const double & factor);

  // getters
  /**
   * This method is kept for compatibility, but it can also be used to get one particular
   * set of gain values. By default it will return the first gain, so if there is only
   * one it will work like the old version and the old code using it will work.
   * The second parameter allows to specify which gain to retrieve, considering that
   * they are in input order.
   * NOTE that no protection is inside the method (because we want to keep it very light)
   * therefore it is the caller duty to check that the index is in the correct range.
   */
  const SiStripApvGain::Range getRange(const uint32_t& detID, const int index = 0) const;
  /// Returns a vector of ranges for all the gains in the format expected by getStripGain and getApvGain.
  const vector<SiStripApvGain::Range> getAllRanges(const uint32_t& DetId) const;

  /// Used to get the gain of a specific gain set
  float getStripGain(const uint16_t& strip, const SiStripApvGain::Range& range, const int index = 0) const;
  /// Used to get the full gain (product of all the gains)
  float getStripGain(const uint16_t& strip, const vector<SiStripApvGain::Range>& range) const;
  /// Used to get the gain of a specific gain set
  float getApvGain(const uint16_t& apv, const SiStripApvGain::Range& range, const int index = 0) const;
  /// Used to get the full gain (product of all the gains)
  float getApvGain(const uint16_t& apv, const vector<SiStripApvGain::Range>& rangeVector) const;
  /// ATTENTION: we assume the detIds are the same as those from the first gain
  void getDetIds(std::vector<uint32_t>& DetIds_) const;

  void printDebug(std::stringstream& ss) const;
  void printSummary(std::stringstream& ss) const;

 private:

  SiStripGain(const SiStripGain&); // stop default
  const SiStripGain& operator=(const SiStripGain&); // stop default

  // ---------- member data --------------------------------

  vector<const SiStripApvGain *> apvgain_;
  vector<double> norm_;
};

#endif
