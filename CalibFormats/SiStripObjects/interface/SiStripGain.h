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
 * $Id: SiStripGain.h,v 1.6 2010/03/29 12:32:37 demattia Exp $
 *
 * Modifications by M. De Mattia (demattia@pd.infn.it) on 11/11/2009:
 * It now holds a std::vector of pointers to ApvGain and a std::vector of corresponding
 * normalization factors. <br>
 * It returns the product of all the Gain/norm ratios. <br>
 * The multiply method allows to input additional gain records. <br>
 * ATTENTION: we take the list of detIds from the first gain and we assume
 * that is it the same for all the rest of the gains. <br>
 * <br>
 * The getStripGain and getApvGain return the gain value for the selected range of a gain set
 * specified by index in the internal std::vector (default 0). This std::vector is built adding each
 * gain in sequence and retains this order, so the first set of gain input will be 0 ecc... <br>
 * Additional overloaded methods receive the full std::vector of ranges for all the gains and returns
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
#include <memory>

class SiStripGain
{
 public:
  SiStripGain() {};
  virtual ~SiStripGain() {};

  inline SiStripGain(const SiStripApvGain& apvgain, const double & factor) :
    apvgain_(0)
  {
    multiply(apvgain, factor);
  }

  /// Used to input additional gain values that will be multiplied to the first one
  void multiply(const SiStripApvGain & apvgain, const double & factor);

  // getters
  // For the product of all apvGains
  // -------------------------------
  const SiStripApvGain::Range getRange(const uint32_t& detID) const;
  float getStripGain(const uint16_t& strip, const SiStripApvGain::Range& range) const;
  float getApvGain(const uint16_t& apv, const SiStripApvGain::Range& range) const;

  // For a specific apvGain
  // ----------------------
  /**
   * The second parameter allows to specify which gain to retrieve, considering that
   * they are in input order.
   * NOTE that no protection is inside the method (because we want to keep it very light)
   * therefore it is the caller duty to check that the index is in the correct range.
   */
  const SiStripApvGain::Range getRange(const uint32_t& detID, const int index) const;
  float getStripGain(const uint16_t& strip, const SiStripApvGain::Range& range, const int index) const;
  float getApvGain(const uint16_t& apv, const SiStripApvGain::Range& range, const int index) const;

  /// ATTENTION: we assume the detIds are the same as those from the first gain
  void getDetIds(std::vector<uint32_t>& DetIds_) const;

  void printDebug(std::stringstream& ss) const;
  void printSummary(std::stringstream& ss) const;

 private:

  void fillNewGain(const SiStripApvGain * apvgain, const double & factor,
		   const SiStripApvGain * apvgain2 = 0, const double & factor2 = 1.);
  SiStripGain(const SiStripGain&); // stop default
  const SiStripGain& operator=(const SiStripGain&); // stop default

  // ---------- member data --------------------------------

  std::vector<const SiStripApvGain *> apvgainVector_;
  std::vector<double> normVector_;
  const SiStripApvGain * apvgain_;
  std::auto_ptr<SiStripApvGain> apvgainAutoPtr_;
};

#endif
