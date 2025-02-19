#include "CondFormats/SiStripObjects/interface/FedTimingAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace sistrip;

// ----------------------------------------------------------------------------
// 
FedTimingAnalysis::FedTimingAnalysis( const uint32_t& key ) 
  : CommissioningAnalysis(key,"FedTimingAnalysis"),
    time_(sistrip::invalid_), 
    max_(sistrip::invalid_), 
    delay_(sistrip::invalid_), 
    error_(sistrip::invalid_), 
    base_(sistrip::invalid_), 
    peak_(sistrip::invalid_), 
    height_(sistrip::invalid_),
    optimumSamplingPoint_(15.)
{;}

// ----------------------------------------------------------------------------
// 
FedTimingAnalysis::FedTimingAnalysis() 
  : CommissioningAnalysis("FedTimingAnalysis"),
    time_(sistrip::invalid_), 
    max_(sistrip::invalid_), 
    delay_(sistrip::invalid_), 
    error_(sistrip::invalid_), 
    base_(sistrip::invalid_), 
    peak_(sistrip::invalid_), 
    height_(sistrip::invalid_),
    optimumSamplingPoint_(15.)
{;}

// ----------------------------------------------------------------------------
// 
void FedTimingAnalysis::reset() {
  time_ = sistrip::invalid_; 
  max_ = sistrip::invalid_; 
  delay_ = sistrip::invalid_; 
  error_ = sistrip::invalid_; 
  base_ = sistrip::invalid_; 
  peak_ = sistrip::invalid_; 
  height_ = sistrip::invalid_;
}

// ----------------------------------------------------------------------------
// 
void FedTimingAnalysis::print( std::stringstream& ss, uint32_t not_used ) { 
  header( ss );
  ss << " Time of tick rising edge [ns]      : " << time_ << "\n" 
     << " Maximum time (sampling point) [ns] : " << max_ << "\n" 
     << " Delay required wrt max time [ns]   : " << delay_ << "\n" 
     << " Error on delay [ns]                : " << error_ << "\n"
     << " Baseline [adc]                     : " << base_ << "\n" 
     << " Tick peak [adc]                    : " << peak_ << "\n" 
     << " Tick height [adc]                  : " << height_ << "\n";
}

// ----------------------------------------------------------------------------
// 
void FedTimingAnalysis::max( const float& max ) { 
  max_ = max;
  if ( time_ > sistrip::maximum_ ) { return; }
  int32_t adjustment = 25 - static_cast<int32_t>( rint( max_ + optimumSamplingPoint_ ) ) % 25;
  max_ += adjustment;
  delay_ = max_ - time_; 
}
