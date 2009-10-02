 // -*- C++ -*-
//
// Package:     SiStripObjects
// Class  :     SiStripGain
// Implementation:
//     <Notes on implementation>
// Original Author:  gbruno
//         Created:  Wed Mar 22 12:24:33 CET 2006
// $Id: SiStripGain.cc,v 1.4 2009/03/27 18:48:58 giordano Exp $

#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "sstream"

using namespace std;


SiStripGain::SiStripGain()
{
}



SiStripGain::~SiStripGain()
{
}


SiStripGain::SiStripGain(const SiStripApvGain& apvgain, double factor) : apvgain_(&apvgain), norm_(factor)
{

}

float SiStripGain::getStripGain(const uint16_t& strip, const SiStripApvGain::Range& range) const{

  return (apvgain_->getStripGain(strip,range))/norm_;

}
float SiStripGain::getApvGain(const uint16_t& apv, const SiStripApvGain::Range& range) const {

  return (apvgain_->getApvGain(apv,range))/norm_;

}


void SiStripGain::getDetIds(std::vector<uint32_t>& DetIds_) const {

  return apvgain_->getDetIds(DetIds_);

}

const SiStripApvGain::Range SiStripGain::getRange(const uint32_t& DetId) const {
  return apvgain_->getRange(DetId);

}


void SiStripGain::printDebug(std::stringstream& ss) const {

  std::vector<uint32_t> detid;
  getDetIds(detid);

  ss << "detid \t|\t apv lists\n"; 
  for (size_t id=0;id<detid.size();id++){
    SiStripApvGain::Range range=getRange(detid[id]);
    ss  <<  detid[id] << " \t|\t";
    for(int it=0;it<range.second-range.first;it++){
      ss << getApvGain(it,range)     << " \t|\t"; 
    } 
    ss<< std::endl;          
  }
}

void SiStripGain::printSummary(std::stringstream& ss) const{ 
  ss << "SiStripGain::printSummary has to be implemented " << std::endl;
}
