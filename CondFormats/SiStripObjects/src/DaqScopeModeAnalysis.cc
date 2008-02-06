#include "CondFormats/SiStripObjects/interface/DaqScopeModeAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TH1.h"
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
void DaqScopeModeAnalysis::print( std::stringstream& ss, uint32_t not_used ) { 
  header( ss );
  ss << " Number of entries   : " << entries_ << "\n" 
     << " Mean +/- rms [adc]  : " << mean_ << " +/- " << rms_ << "\n"
     << " Median / mode [adc] : " << median_ << " / " << mode_ << "\n" 
     << " Min / max [adc]     : " << min_ << " / " << max_ << "\n";
}

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
  histo_ = Histo(0,"");
}

// ----------------------------------------------------------------------------
// 
void DaqScopeModeAnalysis::extract( const std::vector<TH1*>& histos ) { 
  
  // Check
  if ( histos.size() != 1 ) {
    edm::LogWarning(mlCommissioning_) 
      << "[" << myName() << "::" << __func__ << "]"
      << " Unexpected number of histograms: " 
      << histos.size();
  }
  
  // Extract FED key from histo title
  if ( !histos.empty() ) extractFedKey( histos.front() );

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
    if ( title.runType() != sistrip::DAQ_SCOPE_MODE ) {
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
void DaqScopeModeAnalysis::analyse() { 
  if ( !histo_.first ) {
    edm::LogWarning(mlCommissioning_)
      << "[" << myName() << "::" << __func__ << "]"
      << " NULL pointer to histogram!";
    return;
  }

  // Some initialization
  reset();
  std::vector<float> median;
  float max_value = -1. * sistrip::invalid_;
  float max_contents = -1. * sistrip::invalid_;
  float sum = 0.;
  float sum2 = 0.;
  
  // Entries, min, mode
  uint16_t nbins = static_cast<uint16_t>( histo_.first->GetNbinsX() );
  for ( uint16_t ibin = 0; ibin < nbins; ibin++ ) {
    float value = histo_.first->GetBinLowEdge(ibin+1) + histo_.first->GetBinWidth(ibin+1) / 2.;
    float contents = histo_.first->GetBinContent(ibin+1);
    //float errors = histo_.first->GetBinError(ibin+1);
    if ( contents ) { 
      if ( contents > max_contents ) { mode_ = contents; max_contents = contents; }
      if ( value > max_value ) { max_value = value; }
      if ( value < min_ ) { min_ = value; }
      sum += value * contents;
      sum2 += value * contents* value * contents;
      median.insert( median.end(), static_cast<uint32_t>(contents), value );
    }
    entries_ += contents;
  }
  
  // Max
  if ( max_value > -1. * sistrip::maximum_ ) { max_ = max_value; }

  // Median
  sort( median.begin(), median.end() );
  if ( !median.empty() ) { median_ = median[ median.size()%2 ? median.size()/2 : median.size()/2 ]; }
  
  // Mean, rms
  if ( entries_ ) { 
    sum /= static_cast<float>(entries_);
    sum2 /= static_cast<float>(entries_);
    mean_ = sum;
    if (  sum2 > sum*sum ) { rms_ = sqrt( sum2 - sum*sum ); }
  }
  
}
