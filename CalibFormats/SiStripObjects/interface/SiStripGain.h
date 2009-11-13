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
 * $Id: SiStripGain.h,v 1.3 2009/03/27 18:48:57 giordano Exp $
 *
 * Modifications by M. De Mattia (demattia@pd.infn.it) on 11/11/2009:
 * It now hold a vector of pointers to ApvGain and a vector of corresponding
 * normalization factors. <br>
 * It returns the product of all the Gain/norm ratios. <br>
 * The multiply method allows to input additional gain records.
 * ATTENTION: we make the following assumptions:
 * - we assume the detIds are the ones from the first gain
 * - we assume the range is the one from the first gain
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
  const SiStripApvGain::Range getRange(const uint32_t& detID) const;
  
  float getStripGain(const uint16_t& strip, const SiStripApvGain::Range& range) const;
  float getApvGain(const uint16_t& apv, const SiStripApvGain::Range& range) const;
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
