#include "DQM/SiStripCommissioningAnalysis/interface/ApvTimingAnalysis.h"
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
ApvTimingAnalysis::ApvTimingAnalysis( const uint32_t& key ) 
  : CommissioningAnalysis(key,"ApvTimingAnalysis"),
    time_(sistrip::invalid_), 
    maxTime_(sistrip::invalid_), 
    delay_(sistrip::invalid_), 
    error_(sistrip::invalid_), 
    base_(sistrip::invalid_), 
    peak_(sistrip::invalid_), 
    height_(sistrip::invalid_),
    histo_(0,""),
    optimumSamplingPoint_(15.)
{;}
// ----------------------------------------------------------------------------
// 
ApvTimingAnalysis::ApvTimingAnalysis() 
  : CommissioningAnalysis("ApvTimingAnalysis"),
    time_(sistrip::invalid_), 
    maxTime_(sistrip::invalid_), 
    delay_(sistrip::invalid_), 
    error_(sistrip::invalid_), 
    base_(sistrip::invalid_), 
    peak_(sistrip::invalid_), 
    height_(sistrip::invalid_),
    histo_(0,""),
    optimumSamplingPoint_(15.)
{;}

// ----------------------------------------------------------------------------
// 
void ApvTimingAnalysis::print( std::stringstream& ss, uint32_t not_used ) { 
  header( ss );
  float adjust = sistrip::invalid_;
  if ( time_ <= sistrip::maximum_ && 
       delay_ <= sistrip::maximum_ ) { adjust = time_ + delay_; }
  ss << " Time of tick mark rising edge [ns]        : " << time_ << "\n" 
     << " Error on time of rising edge [ns]         : " << error_ << "\n"
     << " Sampling point of last tick mark [ns]     : " << maxTime_ + optimumSamplingPoint_ << "\n" 
     << " Adjusted sampling point of last tick [ns] : " << adjust << "\n" 
     << " Delay required to synchronise [ns]        : " << delay_ << "\n" 
     << " Baseline level [adc]                      : " << base_ << "\n" 
     << " Tick peak level [adc]                     : " << peak_ << "\n" 
     << " Tick mark height [adc]                    : " << height_ << "\n";
}

// ----------------------------------------------------------------------------
// 
void ApvTimingAnalysis::reset() {
  time_ = sistrip::invalid_; 
  maxTime_ = sistrip::invalid_; 
  delay_ = sistrip::invalid_; 
  error_ = sistrip::invalid_; 
  base_ = sistrip::invalid_; 
  peak_ = sistrip::invalid_; 
  height_ = sistrip::invalid_;
  histo_ = Histo(0,"");
}

// ----------------------------------------------------------------------------
// 1) Calculates position of "sampling point" of last tick; 2) then
// adjusts so that it is multiple of 25, ie, synched with FED
// sampling; 3) then calculates delay required to synchronise with
// this adjusted sampling position.
void ApvTimingAnalysis::maxTime( const float& time ) { 
  maxTime_ = time;
  if ( time_ > sistrip::maximum_ ) { return; }
  float adjustment = 25 - static_cast<int32_t>( rint(maxTime_+optimumSamplingPoint_) ) % 25;
  //maxTime_ += adjustment;
  //delay_ = maxTime_ - time_; 
  delay_ = ( maxTime_ + adjustment ) - time_; 
}

// ----------------------------------------------------------------------------
// 
void ApvTimingAnalysis::extract( const std::vector<TH1*>& histos ) { 
  
  // Check
  if ( histos.size() != 1 ) {
    edm::LogWarning(mlCommissioning_) 
      << "[" << myName() << "::" << __func__ << "]"
      << " Unexpected number of histograms: " 
      << histos.size();
  }
  
  // Extract FED key from histo title
  if ( !histos.empty() ) { extractFedKey( histos.front() ); }

  // Extract
  std::vector<TH1*>::const_iterator ihis = histos.begin();
  for ( ; ihis != histos.end(); ihis++ ) {
    
    // Check pointer
    if ( !(*ihis) ) {
      edm::LogWarning(mlCommissioning_) 
	<< "[" << myName() << "::" << __func__ << "]"
	<< " NULL pointer to histogram!";
      continue;
    }
    
    // Check name
    SiStripHistoTitle title( (*ihis)->GetName() );
    if ( title.runType() != sistrip::APV_TIMING ) {
      edm::LogWarning(mlCommissioning_)
	<< "[" << myName() << "::" << __func__ << "]"
	<< " Unexpected commissioning task: "
	<< SiStripEnumsAndStrings::runType(title.runType());
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
    edm::LogWarning(mlCommissioning_) 
      << "[" << myName() << "::" << __func__ << "]"
      << " NULL pointer to histogram!";
    return;
  }

  TProfile* histo = dynamic_cast<TProfile*>(histo_.first);
  if ( !histo ) {
    edm::LogWarning(mlCommissioning_) 
      << "[" << myName() << "::" << __func__ << "]"
      << " NULL pointer to TProfile histogram!";
    return;
  }
  
  // Transfer histogram contents/errors/stats to containers
  uint16_t non_zero = 0;
  float max = -1.e9;
  float min =  1.e9;
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
    edm::LogWarning(mlCommissioning_)
      << "[" << myName() << "::" << __func__ << "]"
      << " Too few bins! Number of bins: " 
      << bin_contents.size();
    return; 
  }
  
  // Calculate range (max-min) and threshold level (range/2)
  float range = max - min;
  float threshold = min + range / 2.;
  if ( range < 50. ) {
    edm::LogWarning(mlCommissioning_) 
      << "[" << myName() << "::" << __func__ << "]"
      << " Device with crate/FEC/Ring/CCU/module/channel "
      << fec().fecCrate() << "/"
      << fec().fecSlot() << "/"
      << fec().fecRing() << "/"
      << fec().ccuAddr() << "/"
      << fec().ccuChan() << "/"
      << fec().channel() 
      << " connected to FED id/ch " 
      << fed().fedId() << "/"
      << fed().fedChannel() 
      << " has a small signal range [ADC] of " 
      << range;
    // Records levels anyway
    base_   = min;
    peak_   = max;
    height_ = max - min;
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
  if ( (tickmark-baseline) < 50. ) {
    edm::LogWarning(mlCommissioning_) 
      << "[" << myName() << "::" << __func__ << "]"
      << " Device with Crate/FEC/Ring/CCU/module/channel "
      << fec().fecCrate() << "/"
      << fec().fecSlot() << "/"
      << fec().fecRing() << "/"
      << fec().ccuAddr() << "/"
      << fec().ccuChan() << "/"
      << fec().channel() 
      << " connected to FED id/ch " 
      << fed().fedId() << "/"
      << fed().fedChannel() 
      << " has a small range [ADC] of " << (tickmark-baseline)
      << " (with the tick mark peak at " << tickmark
      << " and baseline at "  << baseline << ")"; 
    base_   = baseline;
    peak_   = tickmark;
    height_ = tickmark - baseline;
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
  float baseline_rms = 0.;
  if (  mean2 > mean*mean ) { baseline_rms = sqrt( mean2 - mean*mean ); }
  else { baseline_rms = 0.; }
  
  // Find rising edges (derivative across two bins > range/2) 
  std::map<uint16_t,float> edges;
  for ( uint16_t ibin = 1; ibin < nbins-1; ibin++ ) {
    if ( bin_entries[ibin+1] && 
	 bin_entries[ibin-1] ) {
      float derivative = bin_contents[ibin+1] - bin_contents[ibin-1];
      if ( derivative > 5.*baseline_rms ) { 
	edges[ibin] = derivative; 
      }
    }
  }
  
  // Iterate through "edges" std::map
  bool found = false;
  uint16_t deriv_bin = sistrip::invalid_;
  float max_deriv = -1.*sistrip::invalid_;
  std::map<uint16_t,float>::iterator iter = edges.begin();
  while ( !found && iter != edges.end() ) {

    // Iterate through 50 subsequent samples
    bool valid = true;
    for ( uint16_t ii = 0; ii < 50; ii++ ) {
      uint16_t bin = iter->first + ii;

      // Calc local derivative 
      float temp_deriv = 0;
      if ( static_cast<uint32_t>(bin-1) < 0 ||
	   static_cast<uint32_t>(bin+1) >= nbins ) { continue; }
      temp_deriv = bin_contents[bin+1] - bin_contents[bin-1];
      
      // Store max derivative
      if ( temp_deriv > max_deriv ) {
	max_deriv = temp_deriv;
	deriv_bin = bin;
      }

      // Check if samples following edge are all "high"
      if ( ii > 10 && ii < 40 && bin_entries[bin] &&
	   bin_contents[bin] < baseline + 5*baseline_rms ) { valid = false; }

    }

    // Break from loop if tick mark found
    if ( valid ) { found = true; }
    else {
      max_deriv = -1.*sistrip::invalid_;
      deriv_bin = sistrip::invalid_;
      edges.erase(iter);
    }

    iter++;
  }
  
  // RootAnalyzer implementation (not used by default)
  if ( false ) {
    float maxdev = -9999;
    float mindev = 9999;
    int ideriv = 1;
    int idevmin = 1;
    for ( int is = 10; is < histo->GetNbinsX()-10 ; is++ ) {
      float deriv = (histo->GetBinContent(is+1)-histo->GetBinContent(is-1));
      if ( deriv > maxdev ) {
	maxdev=deriv;
	ideriv=is;
      }
      if ( deriv < mindev ) {
	mindev=deriv;
	idevmin=is;
      }
    }
    
    if ( maxdev > 10. ) {
      deriv_bin = ideriv;
      baseline = histo->GetBinContent(ideriv-10);
      tickmark = histo->GetBinContent(ideriv+10);
    } else {
      deriv_bin = 0;
      baseline = 0;
      tickmark = 0;
    }
  }
  
  // Set monitorables
  if ( deriv_bin <= sistrip::maximum_ ) {
    time_      = deriv_bin * 25. / 24.;
    error_     = 0.;
    base_      = baseline;
    peak_      = tickmark;
    height_    = tickmark - baseline;
  } else {
    edm::LogWarning(mlCommissioning_) 
      << "[" << myName() << "::" << __func__ << "]"
      << " No tick marks found!";
    base_   = baseline;
    peak_   = tickmark;
    height_ = tickmark - baseline;
  }
  
}
