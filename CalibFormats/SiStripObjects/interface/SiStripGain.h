#ifndef SiStripObjects_SiStripGain_h
#define SiStripObjects_SiStripGain_h
// -*- C++ -*-
//
// Package:     SiStripObjects
// Class  :     SiStripGain
// 
/**\class SiStripGain SiStripGain.h CalibFormats/SiStripObjects/interface/SiStripGain.h

 Description: give detector view for the cabling classes

 Usage:
    <usage>

*/
//
// Original Author:  gbruno
//         Created:  Wed Mar 22 12:24:20 CET 2006
// $Id: SiStripGain.h,v 1.1 2007/05/02 12:33:33 gbruno Exp $
//

#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"


class SiStripGain
{
public:
  SiStripGain();
  virtual ~SiStripGain();
  SiStripGain(const SiStripApvGain &, double factor);
  
  // getters
  const SiStripApvGain::Range getRange(const uint32_t& detID) const;
  
  float getStripGain(const uint16_t& strip, const SiStripApvGain::Range& range) const;
  float getApvGain(const uint16_t& apv, const SiStripApvGain::Range& range) const;
  void getDetIds(std::vector<uint32_t>& DetIds_) const;

  private:

    SiStripGain(const SiStripGain&); // stop default
    const SiStripGain& operator=(const SiStripGain&); // stop default

    // ---------- member data --------------------------------

    const SiStripApvGain * apvgain_;
    double norm_;

};

#endif
