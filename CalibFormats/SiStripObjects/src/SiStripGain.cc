 // -*- C++ -*-
//
// Package:     SiStripObjects
// Class  :     SiStripGain
// Implementation:
//     <Notes on implementation>
// Original Author:  gbruno
//         Created:  Wed Mar 22 12:24:33 CET 2006
// $Id: SiStripGain.cc,v 1.8 2009/11/16 10:06:30 demattia Exp $

#include "FWCore/Utilities/interface/typelookup.h"
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

float SiStripGain::getStripGain(const uint16_t& strip, const SiStripApvGain::Range& range, const int index) const
{
  if( !(apvgain_.empty()) ) {
    return (apvgain_[index]->getStripGain(strip, range))/(norm_[0]);
  }
  else {
    edm::LogError("SiStripGain::getStripGain") << "ERROR: no gain available. Returning gain = 1." << endl;
    return 1.;
  }
}

float SiStripGain::getStripGain(const uint16_t& strip, const vector<SiStripApvGain::Range>& rangeVector) const
{
  if( !(apvgain_.empty()) ) {
    double gain = 1.;
    vector<SiStripApvGain::Range>::const_iterator range = rangeVector.begin();
    int i = 0;
    for( ; range != rangeVector.end(); ++range, ++i ) {
      gain*=(apvgain_[i]->getStripGain(strip, *range))/(norm_[i]);
    }
    return gain;
  }
  else {
    edm::LogError("SiStripGain::getStripGain") << "ERROR: no gain available. Returning gain = 1." << endl;
    return 1.;
  }
}

float SiStripGain::getApvGain(const uint16_t& apv, const SiStripApvGain::Range& range, const int index) const
{
  if( !(apvgain_.empty()) ) {
    return (apvgain_[index]->getApvGain(apv, range))/(norm_[0]);
  }
  else {
    edm::LogError("SiStripGain::getApvGain") << "ERROR: no gain available. Returning gain = 1." << endl;
    return 1.;
  }
}

float SiStripGain::getApvGain(const uint16_t& apv, const vector<SiStripApvGain::Range>& rangeVector) const
{
  if( !(apvgain_.empty()) ) {
    // cout << "apvgain_.size() = " << apvgain_.size() << endl;
    // cout << "apvgain_[0] = " << apvgain_[0]->getApvGain(apv, range) << endl;
    double gain = 1.;
    vector<SiStripApvGain::Range>::const_iterator range = rangeVector.begin();
    int i = 0;
    for( ; range != rangeVector.end(); ++range, ++i ) {
      gain *= apvgain_[i]->getApvGain(apv, *range)/norm_[i];
      // cout << "apvgain_["<<i<<"] = " << apvgain_[i]->getApvGain(apv, *range) << endl;
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
  // ATTENTION: we assume the detIds are the same as those from the first gain
  return apvgain_[0]->getDetIds(DetIds_);
}

const SiStripApvGain::Range SiStripGain::getRange(const uint32_t& DetId, const int index) const
{
  return apvgain_[index]->getRange(DetId);
}

const vector<SiStripApvGain::Range> SiStripGain::getAllRanges(const uint32_t& DetId) const
{
  vector<SiStripApvGain::Range> allRanges;
  vector<const SiStripApvGain*>::const_iterator apvGainIt = apvgain_.begin();
  for( ; apvGainIt != apvgain_.end(); ++apvGainIt ) {
    allRanges.push_back((*apvGainIt)->getRange(DetId));
  }
  return allRanges;
}

void SiStripGain::printDebug(std::stringstream & ss) const
{
  vector<unsigned int> detIds;
  getDetIds(detIds);
  std::vector<unsigned int>::const_iterator detid = detIds.begin();
  ss << "Number of detids " << detIds.size() << std::endl;

  for( ; detid != detIds.end(); ++detid ) {
    vector<SiStripApvGain::Range> rangeVector = getAllRanges(*detid);
    if( !rangeVector.empty() ) {
      // SiStripApvGain::Range range = getRange(*detid);
      int apv=0;
      for( int it=0; it < rangeVector[0].second - rangeVector[0].first; ++it ) {
        ss << "detid " << *detid << " \t"
           << " apv " << apv++ << " \t"
           << getApvGain(it,rangeVector) << " \t"
           << std::endl;
      }
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
    vector<SiStripApvGain::Range> rangeVector = getAllRanges(*detid);
    if( !rangeVector.empty() ) {
      for( int it=0; it < rangeVector[0].second - rangeVector[0].first; ++it ) {
        summaryGain.add(*detid, getApvGain(it, rangeVector));
      }
    }
  }
  ss << "Summary of gain values:" << endl;
  summaryGain.print(ss, true);
}
