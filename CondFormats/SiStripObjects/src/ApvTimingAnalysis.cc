#include "CondFormats/SiStripObjects/interface/ApvTimingAnalysis.h"
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
const float ApvTimingAnalysis::optimumSamplingPoint_ = 15.; // [ns]

// ----------------------------------------------------------------------------
// 
const float ApvTimingAnalysis::tickMarkHeightThreshold_ = 50.; // [ADC]

// ----------------------------------------------------------------------------
// 
const float ApvTimingAnalysis::frameFindingThreshold_ = (2./3.); // fraction of tick mark height

// ----------------------------------------------------------------------------
// 
float ApvTimingAnalysis::refTime_ = 1.*sistrip::invalid_;

// ----------------------------------------------------------------------------
// 
ApvTimingAnalysis::ApvTimingAnalysis( const uint32_t& key ) 
  : CommissioningAnalysis(key,sistrip::apvTimingAnalysis_),
    time_(1.*sistrip::invalid_), 
    error_(1.*sistrip::invalid_), 
    delay_(1.*sistrip::invalid_), 
    height_(1.*sistrip::invalid_),
    base_(1.*sistrip::invalid_), 
    peak_(1.*sistrip::invalid_), 
    synchronized_(false)
{;}

// ----------------------------------------------------------------------------
// 
ApvTimingAnalysis::ApvTimingAnalysis() 
  : CommissioningAnalysis(sistrip::apvTimingAnalysis_),
    time_(1.*sistrip::invalid_), 
    error_(1.*sistrip::invalid_), 
    delay_(1.*sistrip::invalid_), 
    height_(1.*sistrip::invalid_),
    base_(1.*sistrip::invalid_), 
    peak_(1.*sistrip::invalid_), 
    synchronized_(false)
{;}

// ----------------------------------------------------------------------------
// 
void ApvTimingAnalysis::reset() {
  time_   = 1.*sistrip::invalid_; 
  error_  = 1.*sistrip::invalid_; 
  delay_  = 1.*sistrip::invalid_; 
  height_ = 1.*sistrip::invalid_;
  base_   = 1.*sistrip::invalid_; 
  peak_   = 1.*sistrip::invalid_; 
  synchronized_ = false;
}

// ----------------------------------------------------------------------------
// 
void ApvTimingAnalysis::refTime( const float& time ) { 

  // Checks synchronization to reference time is done only once
  if ( synchronized_ ) { 
    edm::LogWarning(mlCommissioning_)
      << "[" << myName() << "::" << __func__ << "]"
      << " Attempting to re-synchronize with reference time!"
      << " Not allowed!";
    return; 
  }
  synchronized_ = true;

  // Set reference time and check if tick mark time is valid
  refTime_ = time;
  if ( time_ > sistrip::valid_ ) { return; }
  
  // Calculate position of "sampling point" of last tick;
  int32_t position = static_cast<int32_t>( rint( refTime_ + optimumSamplingPoint_ ) );

  // Calculate adjustment so that sampling point is multiple of 25 (ie, synched with FED sampling)
  float adjustment = 25 - position % 25;

  // Calculate delay required to synchronise with this adjusted sampling position
  delay_ = ( refTime_ + adjustment ) - time_; 

  // Check reference time
  if ( refTime_ < 0. || refTime_ > sistrip::valid_ ) { 
    addErrorCode(sistrip::invalidRefTime_);
  }
  
  // Check delay is valid
  if ( delay_ < 0. || delay_ > sistrip::valid_ ) { 
    addErrorCode(sistrip::invalidDelayTime_);
  }

}

// ----------------------------------------------------------------------------
// 
uint32_t ApvTimingAnalysis::frameFindingThreshold() const { 
  if ( base_ < sistrip::valid_ &&
       peak_ < sistrip::valid_ &&
       height_ > tickMarkHeightThreshold_ ) { 
    float temp1 = base() + height() * ApvTimingAnalysis::frameFindingThreshold_;
    uint32_t temp2 = static_cast<uint32_t>( temp1 );
    return ((temp2/32)*32);
  } else { return sistrip::invalid_; }
}

// ----------------------------------------------------------------------------
// 
bool ApvTimingAnalysis::isValid() const {
  return ( time_    < sistrip::valid_ &&
	   refTime_ < sistrip::valid_ && 
	   // ( !synchronized_ || refTime_ < sistrip::valid_ ) && //@@ ignore validity if not synchronized (ie, set)
	   delay_   < sistrip::valid_ &&
	   height_  < sistrip::valid_ && 
	   base_    < sistrip::valid_ &&
	   peak_    < sistrip::valid_ &&
	   getErrorCodes().empty() );
} 

// ----------------------------------------------------------------------------
// 
void ApvTimingAnalysis::print( std::stringstream& ss, uint32_t not_used ) { 
  header( ss );

  float sampling1 = sistrip::invalid_;
  if ( time_ <= sistrip::valid_ ) { sampling1 = time_ + optimumSamplingPoint_; }
  
  float sampling2 = sistrip::invalid_;
  if ( refTime_ <= sistrip::valid_ ) { sampling2 = refTime_ + optimumSamplingPoint_; }
  
  float adjust = sistrip::invalid_;
  if ( sampling1 <= sistrip::valid_ && delay_ <= sistrip::valid_ ) { adjust = sampling1 + delay_; }

  ss <<  std::fixed << std::setprecision(2)
     << " Tick mark: time of rising edge     [ns] : " << time_ << std::endl 
    //<< " Error on time of rising edge     [ns] : " << error_ << std::endl
     << " Last tick: time of rising edge     [ns] : " << refTime_ << std::endl 
     << " Tick mark: time of sampling point  [ns] : " << sampling1 << std::endl 
     << " Last tick: time of sampling point  [ns] : " << sampling2 << std::endl 
     << " Last tick: adjusted sampling point [ns] : " << adjust << std::endl 
     << " Delay required to synchronise      [ns] : " << delay_ << std::endl 
     << " Tick mark bottom (baseline)       [ADC] : " << base_ << std::endl 
     << " Tick mark top                     [ADC] : " << peak_ << std::endl 
     << " Tick mark height                  [ADC] : " << height_ << std::endl
     << std::boolalpha 
     << " isValid                                 : " << isValid()  << std::endl
     << std::noboolalpha
     << " Error codes (found "
     << std::setw(2) << std::setfill(' ') << getErrorCodes().size() 
     << ")                  : ";
  if ( getErrorCodes().empty() ) { ss << "(none)"; }
  else { 
    VString::const_iterator istr = getErrorCodes().begin();
    VString::const_iterator jstr = getErrorCodes().end();
    for ( ; istr != jstr; ++istr ) { ss << *istr << " "; }
  }
  ss << std::endl;
}

