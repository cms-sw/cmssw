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
 * $Id: SiStripGain.h,v 1.7 2010/04/15 12:47:38 demattia Exp $
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
 * user to not pass an index value that exceeds the number of ApvGains.
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
