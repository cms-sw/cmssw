#include "CondFormats/SiStripObjects/interface/SamplingAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

using namespace sistrip;

// ----------------------------------------------------------------------------
// 
SamplingAnalysis::SamplingAnalysis( const uint32_t& key ) 
  : CommissioningAnalysis(key,"SamplingAnalysis"),
    sOnCut_(3),
    max_(sistrip::invalid_),
    error_(sistrip::invalid_)
{;}

// ----------------------------------------------------------------------------
// 
SamplingAnalysis::SamplingAnalysis() 
  : CommissioningAnalysis("SamplingAnalysis"),
    sOnCut_(3),
    max_(sistrip::invalid_),
    error_(sistrip::invalid_)
{;}

// ----------------------------------------------------------------------------
// 
void SamplingAnalysis::reset() {
  error_ = sistrip::invalid_;
  max_ = sistrip::invalid_;
}

// ----------------------------------------------------------------------------
// 
void SamplingAnalysis::print( std::stringstream& ss, uint32_t not_used ) { 
  header( ss );
  ss << " Granularity: " << SiStripEnumsAndStrings::granularity(granularity_) << std::endl;
  ss << " Delay corresponding to the maximum of the pulse : " << max_ << std::endl
     << " Error on the position (from the fit)            : " << error_ << std::endl;
}

// ----------------------------------------------------------------------------
//
float SamplingAnalysis::limit(float SoNcut) const
{
  return 3.814567e+00+8.336601e+00*SoNcut-1.511334e-01*pow(SoNcut,2);
}

// ----------------------------------------------------------------------------
//
float SamplingAnalysis::correctMeasurement(float mean, float SoNcut) const
{
  if(mean>limit(SoNcut))
    return -8.124872e+00+9.860108e-01*mean-3.618158e-03*pow(mean,2)+2.037263e-05*pow(mean,3);
  else return 0.;
}
