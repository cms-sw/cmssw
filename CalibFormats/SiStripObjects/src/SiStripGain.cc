 // -*- C++ -*-
//
// Package:     SiStripObjects
// Class  :     SiStripGain
// Implementation:
//     <Notes on implementation>
// Original Author:  gbruno
//         Created:  Wed Mar 22 12:24:33 CET 2006
// $Id: SiStripGain.cc,v 1.7 2007/02/15 11:22:38 gbruno Exp $

#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
//#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;


SiStripGain::SiStripGain()
{
}



SiStripGain::~SiStripGain()
{
}


SiStripGain::SiStripGain(const SiStripApvGain& apvgain) : apvgain_(&apvgain)
{

}

float SiStripGain::getStripGain(const uint16_t& strip, const SiStripApvGain::Range& range) const{

  return apvgain_->getStripGain(strip,range);

}
float SiStripGain::getApvGain(const uint16_t& apv, const SiStripApvGain::Range& range) const {

  return apvgain_->getApvGain(apv,range);

}


void SiStripGain::getDetIds(std::vector<uint32_t>& DetIds_) const {

  return apvgain_->getDetIds(DetIds_);

}

const SiStripApvGain::Range SiStripGain::getRange(const uint32_t& DetId) const {
  return apvgain_->getRange(DetId);

}



EVENTSETUP_DATA_REG(SiStripGain);
