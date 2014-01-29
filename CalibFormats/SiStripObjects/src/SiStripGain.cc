 // -*- C++ -*-
//
// Package:     SiStripObjects
// Class  :     SiStripGain
// Implementation:
//     <Notes on implementation>
// Original Author:  gbruno
//         Created:  Wed Mar 22 12:24:33 CET 2006

#include "FWCore/Utilities/interface/typelookup.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include <sstream>

void SiStripGain::multiply(const SiStripApvGain & apvgain, const double & factor,
			   const std::pair<std::string, std::string> & recordLabelPair)
{
  // When inserting the first ApvGain
  if( apvgain_ == 0 ) {
    if( (factor != 1) && (factor != 0) ) {
      fillNewGain( &apvgain, factor );
    }
    else {
      // If the normalization factor is one, no need to create a new SiStripApvGain
      apvgain_ = &apvgain;
    }
  }
  else {
    // There is already an ApvGain inside the SiStripGain. Multiply it by the new one and save the new pointer.
    fillNewGain( apvgain_, 1., &apvgain, factor ); 
  }
  recordLabelPair_.push_back(recordLabelPair);
  apvgainVector_.push_back(&apvgain);
  normVector_.push_back(factor);
}

void SiStripGain::fillNewGain(const SiStripApvGain * apvgain, const double & factor,
		 const SiStripApvGain * apvgain2, const double & factor2)
{
  SiStripApvGain * newApvGain = new SiStripApvGain;
  edm::FileInPath fp("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat");
  SiStripDetInfoFileReader reader(fp.fullPath());
  const std::map<uint32_t, SiStripDetInfoFileReader::DetInfo> DetInfos = reader.getAllData();

  // Loop on the apvgain in input and fill the newApvGain with the values/factor.
  std::vector<uint32_t> detIds;
  apvgain->getDetIds(detIds);
  std::vector<uint32_t>::const_iterator it = detIds.begin();
  for( ; it != detIds.end(); ++it ) {

    std::map<uint32_t, SiStripDetInfoFileReader::DetInfo>::const_iterator detInfoIt = DetInfos.find(*it);
    if( detInfoIt != DetInfos.end() ) {

      std::vector<float> theSiStripVector;

      // Loop on all the apvs and then on the strips
      SiStripApvGain::Range range = apvgain->getRange(*it);

      SiStripApvGain::Range range2;
      if( apvgain2 != 0 ) {
	range2 = apvgain2->getRange(*it);
      }

      for( int apv = 0; apv < detInfoIt->second.nApvs; ++apv ) {
	float apvGainValue = apvgain->getApvGain( apv, range )/factor;

	if( (apvgain2 != 0) && (factor2 != 0.) ) {
	  apvGainValue *= apvgain2->getApvGain( apv, range2 )/factor2;
	}

	theSiStripVector.push_back(apvGainValue);
      }
      SiStripApvGain::Range inputRange(theSiStripVector.begin(), theSiStripVector.end());
      if( ! newApvGain->put(*it, inputRange) ) {
	edm::LogError("SiStripGain") << "detid already exists" << std::endl;
      }
    }
  }
  apvgain_ = newApvGain;
  // Deletes the managed object and replaces it with the new one
  apvgainAutoPtr_.reset(newApvGain);
}



float SiStripGain::getStripGain(const uint16_t& strip, const SiStripApvGain::Range& range, const uint32_t index) const
{
  if( !(apvgainVector_.empty()) ) {
    return( apvgainVector_[index]->getStripGain(strip, range) );
  }
  edm::LogError("SiStripGain::getStripGain") << "ERROR: no gain available. Returning gain = 1." << std::endl;
  return 1.;
}



float SiStripGain::getApvGain(const uint16_t& apv, const SiStripApvGain::Range& range, const uint32_t index) const
{
  if( !(apvgainVector_.empty()) ) {
    return (apvgainVector_[index]->getApvGain(apv, range))/(normVector_[index]);
  }
  edm::LogError("SiStripGain::getApvGain") << "ERROR: no gain available. Returning gain = 1." << std::endl;
  return 1.;
}

void SiStripGain::getDetIds(std::vector<uint32_t>& DetIds_) const
{
  // ATTENTION: we assume the detIds are the same as those from the first gain
  return apvgain_->getDetIds(DetIds_);
}


const SiStripApvGain::Range SiStripGain::getRange(const uint32_t& DetId, const uint32_t index) const
{
  return apvgainVector_[index]->getRange(DetId);
}

void SiStripGain::printDebug(std::stringstream & ss) const
{
  std::vector<unsigned int> detIds;
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

  std::vector<unsigned int> detIds;
  getDetIds(detIds);
  std::vector<uint32_t>::const_iterator detid = detIds.begin();
  for( ; detid != detIds.end(); ++detid ) {
    SiStripApvGain::Range range = getRange(*detid);
    for( int it=0; it < range.second - range.first; ++it ) {
      summaryGain.add(*detid, getApvGain(it, range));
    }
  }
  ss << "Summary of gain values:" << std::endl;
  summaryGain.print(ss, true);
}
