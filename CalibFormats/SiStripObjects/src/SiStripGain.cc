 // -*- C++ -*-
//
// Package:     SiStripObjects
// Class  :     SiStripGain
// Implementation:
//     <Notes on implementation>
// Original Author:  gbruno
//         Created:  Wed Mar 22 12:24:33 CET 2006
// $Id: SiStripGain.cc,v 1.6 2009/05/08 08:26:10 demattia Exp $

#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"
#include "sstream"

using namespace std;

void SiStripGain::multiply(const SiStripApvGain & apvgain, const double & factor)
{
  apvgain_.push_back(&apvgain);
  norm_.push_back(factor);
}

float SiStripGain::getStripGain(const uint16_t& strip, const SiStripApvGain::Range& range) const
{
  if( !(apvgain_.empty()) ) {

    double gain = (apvgain_[0]->getStripGain(strip, range))/(norm_[0]);

    if( apvgain_.size() > 1 ) {
      // Start looping from the second
      vector<const SiStripApvGain*>::const_iterator apvIt = apvgain_.begin()+1;
      vector<double>::const_iterator normIt = norm_.begin()+1;
      for( ; apvIt != apvgain_.end(); ++apvIt, ++normIt ) {
        gain*=((*apvIt)->getStripGain(strip, range))/(*normIt);
      }
    }

    return gain;
  }
  else {
    edm::LogError("SiStripGain::getStripGain") << "ERROR: no gain available. Returning gain = 1." << endl;
    return 1.;
  }
}

float SiStripGain::getApvGain(const uint16_t& apv, const SiStripApvGain::Range& range) const
{
  if( !(apvgain_.empty()) ) {

    double gain = (apvgain_[0]->getApvGain(apv, range))/(norm_[0]);

    if( apvgain_.size() > 1 ) {
      // Start looping from the second
      vector<const SiStripApvGain*>::const_iterator apvIt = apvgain_.begin()+1;
      vector<double>::const_iterator normIt = norm_.begin()+1;
      for( ; apvIt != apvgain_.end(); ++apvIt, ++normIt ) {
        gain*=((*apvIt)->getApvGain(apv, range))/(*normIt);
      }
    }

    return gain;
  }
  else {
    edm::LogError("SiStripGain::getApvGain") << "ERROR: no gain available. Returning gain = 1." << endl;
    return 1.;
  }
}

void SiStripGain::getDetIds(std::vector<uint32_t>& DetIds_) const
{
  // Note: we assume the detIds are the ones from the first gain
  return apvgain_[0]->getDetIds(DetIds_);
}

const SiStripApvGain::Range SiStripGain::getRange(const uint32_t& DetId) const
{
  // Note: we assume the range is the one from the first gain
  return apvgain_[0]->getRange(DetId);
}

void SiStripGain::printDebug(std::stringstream& ss) const
{
  vector<const SiStripApvGain*>::const_iterator apvIt = apvgain_.begin();
  for( ; apvIt != apvgain_.end(); ++apvIt ) {
    (*apvIt)->printDebug(ss);
  }
}

void SiStripApvGain::printDebug(std::stringstream & ss) const
{
  vector<unsigned int> detIds;
  getDetIds(detIds);
  std::vector<unsigned int>::const_iterator detid = detIds.begin();
  ss << "Number of detids " << detIds.size() << std::endl;

  for( ; detid != detIds.end(); ++detid ) {
    SiStripApvGain::Range range = getRange(*detid);
    int apv=0;
    for( int it=0; it < range.second - range.first; ++it ) {
      ss << "detid " << *detid << " \t"
         << " apv " << apv++ << " \t"
         << getApvGain(it,range) << " \t"
         << std::endl;
    }
  }
}

void SiStripGain::printSummary(std::stringstream& ss) const
{
  SiStripDetSummary summaryGain;

  vector<unsigned int> detIds;
  getDetIds(detIds);
  std::vector<uint32_t>::const_iterator detid = detIds.begin();
  for( ; detid != detIds.end(); ++detid ) {
    SiStripApvGain::Range range = getRange(*detid);
    for( int it=0; it < range.second - range.first; ++it ) {
      summaryGain.add(*detid, getApvGain(it, range));
    }
  }
  ss << "Summary of gain values:" << endl;
  summaryGain.print(ss, true);
}
