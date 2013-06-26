#include "CondFormats/SiStripObjects/interface/FedCablingAnalysis.h"
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
const float FedCablingAnalysis::threshold_ = 100.; // [ADC]

// ----------------------------------------------------------------------------
// 
FedCablingAnalysis::FedCablingAnalysis( const uint32_t& key ) 
  : CommissioningAnalysis(key,sistrip::fedCablingAnalysis_),
    fedId_(sistrip::invalid_), 
    fedCh_(sistrip::invalid_),
    adcLevel_(1.*sistrip::invalid_),
    candidates_()
{;}

// ----------------------------------------------------------------------------
// 
FedCablingAnalysis::FedCablingAnalysis() 
  : CommissioningAnalysis(sistrip::fedCablingAnalysis_),
    fedId_(sistrip::invalid_), 
    fedCh_(sistrip::invalid_),
    adcLevel_(1.*sistrip::invalid_),
    candidates_()
{;}

// ----------------------------------------------------------------------------
// 
void FedCablingAnalysis::reset() {
    fedId_ = sistrip::invalid_; 
    fedCh_ = sistrip::invalid_;
    adcLevel_ = 1.*sistrip::invalid_;
    candidates_.clear();
}

// ----------------------------------------------------------------------------
// 
bool FedCablingAnalysis::isValid() const {
  return ( fedId_ < sistrip::maximum_ &&
	   fedCh_ < sistrip::maximum_ &&
	   adcLevel_ < 1+sistrip::maximum_ &&
	   !candidates_.empty() &&
	   getErrorCodes().empty() );
} 

// ----------------------------------------------------------------------------
// 
void FedCablingAnalysis::print( std::stringstream& ss, uint32_t not_used ) { 
  header( ss );
  ss <<  std::fixed << std::setprecision(2)
     << " Connected FED id               : " << fedId_ << std::endl 
     << " Connected FED channel          : " << fedCh_ << std::endl
     << " Signal level             [ADC] : " << adcLevel_ << std::endl;
  ss << " nCandidates                    : " << candidates_.size() << std::endl
     << " FedId/FedChannel/ADC) : ";
  if ( candidates_.empty() ) { ss << "(none)"; }
  else {
    Candidates::const_iterator iter;
    for ( iter = candidates_.begin(); iter != candidates_.end(); iter++ ) { 
      SiStripFedKey path( iter->first );
      ss << path.fedId() << "/" 
	 << path.fedChannel() << "/" 
	 << iter->second << " ";
    }
  }
  ss << std::endl;
  ss << std::boolalpha
     << " isValid                : " << isValid()  << std::endl
     << std::noboolalpha
     << " Error codes (found "  
     << std::setw(2) << std::setfill(' ') << getErrorCodes().size() 
     << ") : ";
  if ( getErrorCodes().empty() ) { ss << "(none)"; }
  else { 
    VString::const_iterator istr = getErrorCodes().begin();
    VString::const_iterator jstr = getErrorCodes().end();
    for ( ; istr != jstr; ++istr ) { ss << *istr << " "; }
  }
  ss << std::endl;
}
