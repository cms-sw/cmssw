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
void ApvTimingAnalysis::refTime( const float& time, const float& targetDelay ) { 

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
  int32_t position;
  if ( targetDelay == -1 ) {   // by default use latest tick
    position = static_cast<int32_t>( rint( refTime_ + optimumSamplingPoint_ ) );
  } else {
    position = static_cast<int32_t>( rint( targetDelay + optimumSamplingPoint_ ) );
  }

  // Calculate adjustment so that sampling point is multiple of 25 (ie, synched with FED sampling)
  float adjustment = 25 - position % 25;

  // Calculate delay required to synchronise with this adjusted sampling position
  if ( targetDelay == -1 ) {   // by default align forward to the latest tick
    delay_ = ( refTime_ + adjustment ) - time_;
  } else {                     // otherwise use the supplied target delay
    if ( adjustment > 25/2 ) adjustment -= 25; // go as close as possible to desired target
    delay_ = ( targetDelay + adjustment ) - time_;
  }

  // Check reference time
  if ( refTime_ < 0. || refTime_ > sistrip::valid_ ) { 
    refTime_ = sistrip::invalid_;
    addErrorCode(sistrip::invalidRefTime_);
  }
  
  // Check delay is valid
  if ( delay_ < -sistrip::valid_ || delay_ > sistrip::valid_ ) { 
    delay_ = sistrip::invalid_;
    addErrorCode(sistrip::invalidDelayTime_);
  }

}

// ----------------------------------------------------------------------------
// 
uint16_t ApvTimingAnalysis::frameFindingThreshold() const { 
  if ( getErrorCodes().empty() &&
       time_   < sistrip::valid_ &&
       base_   < sistrip::valid_ && 
       peak_   < sistrip::valid_ && 
       height_ < sistrip::valid_  &&
       height_ > tickMarkHeightThreshold_ ) { 
    return ( ( static_cast<uint16_t>( base_ + 
				      height_ * 
				      ApvTimingAnalysis::frameFindingThreshold_ ) / 32 ) * 32 );
  } else { return sistrip::invalid_; }
}

// ----------------------------------------------------------------------------
// 
bool ApvTimingAnalysis::foundTickMark() const {
  return ( getErrorCodes().empty() &&
	   time_   < sistrip::valid_ &&
	   base_   < sistrip::valid_ &&
	   peak_   < sistrip::valid_ &&
	   height_ < sistrip::valid_ &&
	   frameFindingThreshold() < sistrip::valid_ );
} 

// ----------------------------------------------------------------------------
// 
bool ApvTimingAnalysis::isValid() const {
  return ( getErrorCodes().empty() &&
	   time_   < sistrip::valid_ &&
	   base_   < sistrip::valid_ &&
	   peak_   < sistrip::valid_ &&
	   height_ < sistrip::valid_ &&
	   frameFindingThreshold() < sistrip::valid_ &&
	   synchronized_ &&
	   refTime_ < sistrip::valid_ && 
	   delay_   < sistrip::valid_ );
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
     << " Tick mark: time of sampling point  [ns] : " << sampling1 << std::endl 
     << " Ref tick: time of rising edge      [ns] : " << refTime_ << std::endl 
     << " Ref tick: time of sampling point   [ns] : " << sampling2 << std::endl 
     << " Ref tick: adjusted sampling point  [ns] : " << adjust << std::endl 
     << " Delay required to synchronise      [ns] : " << delay_ << std::endl 
     << " Tick mark bottom (baseline)       [ADC] : " << base_ << std::endl 
     << " Tick mark top                     [ADC] : " << peak_ << std::endl 
     << " Tick mark height                  [ADC] : " << height_ << std::endl
     << " Frame finding threshold           [ADC] : " << frameFindingThreshold() << std::endl
     << std::boolalpha 
     << " Tick mark found                         : " << foundTickMark()  << std::endl
     << " isValid                                 : " << isValid()  << std::endl
     << std::noboolalpha
     << " Error codes (found "
     << std::setw(3) << std::setfill(' ') << getErrorCodes().size() 
     << ")                 : ";
  if ( getErrorCodes().empty() ) { ss << "(none)"; }
  else { 
    VString::const_iterator istr = getErrorCodes().begin();
    VString::const_iterator jstr = getErrorCodes().end();
    for ( ; istr != jstr; ++istr ) { ss << *istr << " "; }
  }
  ss << std::endl;
}

