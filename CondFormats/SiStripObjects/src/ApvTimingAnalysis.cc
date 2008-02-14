#include "CondFormats/SiStripObjects/interface/ApvTimingAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TProfile.h"
#include "TH1.h"
#include <iostream>
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
    synchronized_(false),
    histo_(0,"")
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
    synchronized_(false),
    histo_(0,"")
{;}

// ----------------------------------------------------------------------------
// 
void ApvTimingAnalysis::reset() {
  time_ = 1.*sistrip::invalid_; 
  error_ = 1.*sistrip::invalid_; 
  refTime_ = 1.*sistrip::invalid_; 
  delay_ = 1.*sistrip::invalid_; 
  height_ = 1.*sistrip::invalid_;
  base_ = 1.*sistrip::invalid_; 
  peak_ = 1.*sistrip::invalid_; 
  synchronized_ = false;
  histo_ = Histo(0,"");
}

// ----------------------------------------------------------------------------
// 
void ApvTimingAnalysis::extract( const std::vector<TH1*>& histos ) { 
  
  // Check number of histograms
  if ( histos.size() != 1 ) {
    addErrorCode(sistrip::numberOfHistos_);
  }
  
  // Extract FED key from histo title
  if ( !histos.empty() ) { extractFedKey( histos.front() ); }

  // Extract histograms
  std::vector<TH1*>::const_iterator ihis = histos.begin();
  for ( ; ihis != histos.end(); ihis++ ) {
    
    // Check for NULL pointer
    if ( !(*ihis) ) { continue; }
    
    // Check name
    SiStripHistoTitle title( (*ihis)->GetName() );
    if ( title.runType() != sistrip::APV_TIMING ) {
      addErrorCode(sistrip::unexpectedTask_);
      continue;
    }
    
    // Extract timing histo
    histo_.first = *ihis;
    histo_.second = (*ihis)->GetName();
    
  }
  
}

// ----------------------------------------------------------------------------
// 
void ApvTimingAnalysis::analyse() { 

  if ( !histo_.first ) {
    addErrorCode(sistrip::nullPtr_);
    return;
  }
  
  TProfile* histo = dynamic_cast<TProfile*>(histo_.first);
  if ( !histo ) {
    addErrorCode(sistrip::nullPtr_);
    return;
  }
  
  // Transfer histogram contents/errors/stats to containers
  uint16_t non_zero = 0;
  float max = -1. * sistrip::invalid_;
  float min = 1. * sistrip::invalid_;
  uint16_t nbins = static_cast<uint16_t>( histo->GetNbinsX() );
  std::vector<float> bin_contents; 
  std::vector<float> bin_errors;
  std::vector<float> bin_entries;
  bin_contents.reserve( nbins );
  bin_errors.reserve( nbins );
  bin_entries.reserve( nbins );
  for ( uint16_t ibin = 0; ibin < nbins; ibin++ ) {
    bin_contents.push_back( histo->GetBinContent(ibin+1) );
    bin_errors.push_back( histo->GetBinError(ibin+1) );
    bin_entries.push_back( histo->GetBinEntries(ibin+1) );
    if ( bin_entries[ibin] ) { 
      if ( bin_contents[ibin] > max ) { max = bin_contents[ibin]; }
      if ( bin_contents[ibin] < min ) { min = bin_contents[ibin]; }
      non_zero++;
    }
  }
  if ( bin_contents.size() < 100 ) { 
    addErrorCode(sistrip::numberOfBins_);
    return; 
  }
  
  // Calculate range (max-min) and threshold level (range/2)
  float threshold = min + ( max - min ) / 2.;
  base_   = min;
  peak_   = max;
  height_ = max - min;
  if ( max - min < tickMarkHeightThreshold_ ) {
    addErrorCode(sistrip::smallDataRange_);
    return; 
  }
  
  // Associate samples with either "tick mark" or "baseline"
  std::vector<float> tick;
  std::vector<float> base;
  for ( uint16_t ibin = 0; ibin < nbins; ibin++ ) { 
    if ( bin_entries[ibin] ) {
      if ( bin_contents[ibin] < threshold ) { 
	base.push_back( bin_contents[ibin] ); 
      } else { 
	tick.push_back( bin_contents[ibin] ); 
      }
    }
  }
  
  // Find median level of tick mark and baseline
  float tickmark = 0.;
  float baseline = 0.;
  sort( tick.begin(), tick.end() );
  sort( base.begin(), base.end() );
  if ( !tick.empty() ) { tickmark = tick[ tick.size()%2 ? tick.size()/2 : tick.size()/2 ]; }
  if ( !base.empty() ) { baseline = base[ base.size()%2 ? base.size()/2 : base.size()/2 ]; }
  base_   = baseline;
  peak_   = tickmark;
  height_ = tickmark - baseline;
  if ( tickmark - baseline < tickMarkHeightThreshold_ ) {
    addErrorCode(sistrip::smallTickMarkHeight_);
    return; 
  }
  
  // Find rms spread in "baseline" samples
  float mean = 0.;
  float mean2 = 0.;
  for ( uint16_t ibin = 0; ibin < base.size(); ibin++ ) {
    mean += base[ibin];
    mean2 += base[ibin] * base[ibin];
  }
  if ( !base.empty() ) { 
    mean = mean / base.size();
    mean2 = mean2 / base.size();
  } else { 
    mean = 0.; 
    mean2 = 0.; 
  }
  float baseline_rms = sqrt( fabs( mean2 - mean*mean ) ); 

  // Find rising edges (derivative across two bins > threshold) 
  std::map<uint16_t,float> edges;
  for ( uint16_t ibin = 1; ibin < nbins-1; ibin++ ) {
    if ( bin_entries[ibin+1] && 
	 bin_entries[ibin-1] ) {
      float derivative = bin_contents[ibin+1] - bin_contents[ibin-1];
      if ( derivative > 3.*baseline_rms ) { 
	edges[ibin] = derivative; 
      }
    }
  }
  if ( edges.empty() ) {
    addErrorCode(sistrip::noRisingEdges_);
    return;
  }
  
  // Iterate through "edges" map
  uint16_t max_derivative_bin = sistrip::invalid_;
  float max_derivative = -1.*sistrip::invalid_;

  bool found = false;
  std::map<uint16_t,float>::iterator iter = edges.begin();
  while ( !found && iter != edges.end() ) {

    // Iterate through 50 subsequent samples
    bool valid = true;
    for ( uint16_t ii = 0; ii < 50; ii++ ) {
      uint16_t bin = iter->first + ii;

      // Calc local derivative 
      float temp = 0;
      if ( static_cast<uint32_t>(bin-1) < 0 ||
	   static_cast<uint32_t>(bin+1) >= nbins ) { continue; }
      temp = bin_contents[bin+1] - bin_contents[bin-1];
      
      // Store max derivative
      if ( temp > max_derivative ) {
	max_derivative = temp;
	max_derivative_bin = bin;
      }
      
      // Check if samples following edge are all "high"
      if ( ii > 10 && ii < 40 && bin_entries[bin] &&
	   bin_contents[bin] < baseline + 5.*baseline_rms ) { 
	valid = false; 
      }

    }

    // Break from loop if tick mark found
    if ( valid ) { found = true; }
    else {
      max_derivative = -1.*sistrip::invalid_;
      max_derivative_bin = sistrip::invalid_;
      //edges.erase(iter);
      addErrorCode(sistrip::rejectedCandidate_);
    }

    iter++; // next candidate

  }
  
  // Record time monitorable and check tick mark height
  if ( max_derivative_bin <= sistrip::valid_ ) {
    time_ = max_derivative_bin * 25. / 24.;
    if ( height_ < ApvTimingAnalysis::tickMarkHeightThreshold_ ) { 
      addErrorCode(sistrip::tickMarkBelowThresh_);      
    }
  } else {
    addErrorCode(sistrip::missingTickMark_);
  }
  
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

