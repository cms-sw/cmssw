#include "CondFormats/SiStripObjects/interface/ApvLatencyAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <cmath>

using namespace sistrip;

// ----------------------------------------------------------------------------
// 
ApvLatencyAnalysis::ApvLatencyAnalysis( const uint32_t& key ) 
  : CommissioningAnalysis(key,"ApvLatencyAnalysis"),
    latency_(sistrip::invalid_)
{;}

// ----------------------------------------------------------------------------
// 
ApvLatencyAnalysis::ApvLatencyAnalysis() 
  : CommissioningAnalysis("ApvLatencyAnalysis"),
    latency_(sistrip::invalid_)
{;}

// ----------------------------------------------------------------------------
// 
void ApvLatencyAnalysis::reset() {
  latency_ = sistrip::invalid_; 
}

// ----------------------------------------------------------------------------
// 
void ApvLatencyAnalysis::print( std::stringstream& ss, uint32_t not_used ) { 
  header( ss );
  ss << " APV latency setting : " << latency_ << "\n";
}
