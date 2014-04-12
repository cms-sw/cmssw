#include "CondFormats/SiStripObjects/interface/DaqScopeModeAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace sistrip;

// ----------------------------------------------------------------------------
// 
DaqScopeModeAnalysis::DaqScopeModeAnalysis( const uint32_t& key ) 
  : CommissioningAnalysis(key,"DaqScopeModeAnalysis"),
    entries_(sistrip::invalid_), 
    mean_(sistrip::invalid_), 
    median_(sistrip::invalid_), 
    mode_(sistrip::invalid_), 
    rms_(sistrip::invalid_), 
    min_(sistrip::invalid_), 
    max_(sistrip::invalid_)
{;}

// ----------------------------------------------------------------------------
// 
DaqScopeModeAnalysis::DaqScopeModeAnalysis() 
  : CommissioningAnalysis("DaqScopeModeAnalysis"),
    entries_(sistrip::invalid_), 
    mean_(sistrip::invalid_), 
    median_(sistrip::invalid_), 
    mode_(sistrip::invalid_), 
    rms_(sistrip::invalid_), 
    min_(sistrip::invalid_), 
    max_(sistrip::invalid_)
{;}

// ----------------------------------------------------------------------------
// 
void DaqScopeModeAnalysis::reset() {
  entries_ = 1. * sistrip::invalid_; 
  mean_ = 1.*sistrip::invalid_; 
  median_ = 1.*sistrip::invalid_; 
  mode_ = 1.*sistrip::invalid_; 
  rms_ = 1.*sistrip::invalid_; 
  min_ = 1.*sistrip::invalid_; 
  max_ = 1.*sistrip::invalid_; 
}

// ----------------------------------------------------------------------------
// 
void DaqScopeModeAnalysis::print( std::stringstream& ss, uint32_t not_used ) { 
  header( ss );
  ss << " Number of entries   : " << entries_ << "\n" 
     << " Mean +/- rms [adc]  : " << mean_ << " +/- " << rms_ << "\n"
     << " Median / mode [adc] : " << median_ << " / " << mode_ << "\n" 
     << " Min / max [adc]     : " << min_ << " / " << max_ << "\n";
}
