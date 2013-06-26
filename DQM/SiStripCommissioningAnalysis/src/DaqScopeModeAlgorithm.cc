#include "DQM/SiStripCommissioningAnalysis/interface/DaqScopeModeAlgorithm.h"
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
DaqScopeModeAlgorithm::DaqScopeModeAlgorithm( const edm::ParameterSet & pset, DaqScopeModeAnalysis* const anal ) 
  : CommissioningAlgorithm(anal),
    histo_(0,"")
{;}

// ----------------------------------------------------------------------------
// 
void DaqScopeModeAlgorithm::extract( const std::vector<TH1*>& histos ) { 
  
  if ( !anal() ) {
    edm::LogWarning(mlCommissioning_)
      << "[DaqScopeModeAlgorithm::" << __func__ << "]"
      << " NULL pointer to base Analysis object!";
    return; 
  }

  // Check
  if ( histos.size() != 1 ) {
    anal()->addErrorCode(sistrip::numberOfHistos_);
  }
  
  // Extract FED key from histo title
  if ( !histos.empty() ) { anal()->fedKey( extractFedKey( histos.front() ) ); }
  
  // Extract histograms
  std::vector<TH1*>::const_iterator ihis = histos.begin();
  for ( ; ihis != histos.end(); ihis++ ) {
    
    // Check for NULL pointer
    if ( !(*ihis) ) { continue; }
    
    // Check name
    SiStripHistoTitle title( (*ihis)->GetName() );
    if ( title.runType() != sistrip::DAQ_SCOPE_MODE ) {
      anal()->addErrorCode(sistrip::unexpectedTask_);
      continue;
    }
    
    // Extract timing histo
    histo_.first = *ihis;
    histo_.second = (*ihis)->GetName();
    
  }
  
}

// ----------------------------------------------------------------------------
// 
void DaqScopeModeAlgorithm::analyse() { 

  if ( !anal() ) {
    edm::LogWarning(mlCommissioning_)
      << "[DaqScopeModeAlgorithm::" << __func__ << "]"
      << " NULL pointer to base Analysis object!";
    return; 
  }

  CommissioningAnalysis* tmp = const_cast<CommissioningAnalysis*>( anal() );
  DaqScopeModeAnalysis* anal = dynamic_cast<DaqScopeModeAnalysis*>( tmp );
  if ( !anal ) {
    edm::LogWarning(mlCommissioning_)
      << "[DaqScopeModeAlgorithm::" << __func__ << "]"
      << " NULL pointer to derived Analysis object!";
    return; 
  }

  if ( !histo_.first ) {
    anal->addErrorCode(sistrip::nullPtr_);
    return;
  }

  // Some initialization
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
      if ( contents > max_contents ) { anal->mode_ = contents; max_contents = contents; }
      if ( value > max_value ) { max_value = value; }
      if ( value < anal->min_ ) { anal->min_ = value; }
      sum += value * contents;
      sum2 += value * contents* value * contents;
      median.insert( median.end(), static_cast<uint32_t>(contents), value );
    }
    anal->entries_ += contents;
  }
  
  // Max
  if ( max_value > -1. * sistrip::maximum_ ) { anal->max_ = max_value; }

  // Median
  sort( median.begin(), median.end() );
  if ( !median.empty() ) { anal->median_ = median[ median.size()%2 ? median.size()/2 : median.size()/2 ]; }
  
  // Mean, rms
  if ( anal->entries_ ) { 
    sum /= static_cast<float>(anal->entries_);
    sum2 /= static_cast<float>(anal->entries_);
    anal->mean_ = sum;
    if (  sum2 > sum*sum ) { anal->rms_ = sqrt( sum2 - sum*sum ); }
  }
  
}
