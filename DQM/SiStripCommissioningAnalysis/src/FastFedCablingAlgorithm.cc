#include "DQM/SiStripCommissioningAnalysis/interface/FastFedCablingAlgorithm.h"
#include "CondFormats/SiStripObjects/interface/FastFedCablingAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TProfile.h"
#include "TH1.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

using namespace sistrip;

// ----------------------------------------------------------------------------
// 
FastFedCablingAlgorithm::FastFedCablingAlgorithm( const edm::ParameterSet & pset, FastFedCablingAnalysis* const anal ) 
  : CommissioningAlgorithm(anal),
    histo_(nullptr,"")
{;}

// ----------------------------------------------------------------------------
// 
void FastFedCablingAlgorithm::extract( const std::vector<TH1*>& histos ) { 

  if ( !anal() ) {
    edm::LogWarning(mlCommissioning_)
      << "[FastFedCablingAlgorithm::" << __func__ << "]"
      << " NULL pointer to Analysis object!";
    return; 
  }
  
  // Check number of histograms
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
    if ( title.runType() != sistrip::FAST_CABLING ) {
      anal()->addErrorCode(sistrip::unexpectedTask_);
      continue;
    }

    // Extract cabling histo
    histo_.first = *ihis;
    histo_.second = (*ihis)->GetName();
    
  }
  
}

// -----------------------------------------------------------------------------
// 
void FastFedCablingAlgorithm::analyse() { 

  if ( !anal() ) {
    edm::LogWarning(mlCommissioning_)
      << "[FastFedCablingAlgorithm::" << __func__ << "]"
      << " NULL pointer to base Analysis object!";
    return; 
  }

  CommissioningAnalysis* tmp = const_cast<CommissioningAnalysis*>( anal() );
  FastFedCablingAnalysis* anal = dynamic_cast<FastFedCablingAnalysis*>( tmp );
  if ( !anal ) {
    edm::LogWarning(mlCommissioning_)
      << "[FastFedCablingAlgorithm::" << __func__ << "]"
      << " NULL pointer to derived Analysis object!";
    return; 
  }

  if ( !histo_.first ) {
    anal->addErrorCode(sistrip::nullPtr_);
    return;
  }
  
  TProfile* histo = dynamic_cast<TProfile*>(histo_.first);
  if ( !histo ) {
    anal->addErrorCode(sistrip::nullPtr_);
    return;
  }

  // Initialization
  uint16_t zero_entries = 0;
  uint16_t nbins = static_cast<uint16_t>( histo->GetNbinsX() );
  std::vector<float> contents; 
  std::vector<float> errors;
  std::vector<float> entries;
  contents.reserve( nbins );
  errors.reserve( nbins );
  entries.reserve( nbins );

  // Copy histo contents to containers and find min/max
  anal->max_ = -1.*sistrip::invalid_;
  for ( uint16_t ibin = 0; ibin < nbins; ibin++ ) {
    contents.push_back( histo->GetBinContent(ibin+1) );
    errors.push_back( histo->GetBinError(ibin+1) );
    entries.push_back( histo->GetBinEntries(ibin+1) );
    if ( entries[ibin] ) { 
      if ( contents[ibin] > anal->max_ ) { anal->max_ = contents[ibin]; }
      if ( contents[ibin] < anal->min_ ) { anal->min_ = contents[ibin]; }
    } else { zero_entries++; }
  }
  if ( anal->max_ < -1. * sistrip::valid_ ) { anal->max_ = sistrip::invalid_; }
  
  // Check number of bins
  if ( contents.size() != FastFedCablingAnalysis::nBitsForDcuId_ + FastFedCablingAnalysis::nBitsForLldCh_ ) { 
    anal->addErrorCode(sistrip::numberOfBins_);
    return; 
  }
  
  // Check for bins with zero entries
  if ( zero_entries ) { 
    anal->addErrorCode(sistrip::noEntries_);
    return; 
  }

  // Check min and max found
  if ( anal->max_ > sistrip::valid_  || 
       anal->min_ > sistrip::valid_ ) { 
    return; 
  }
  
  // Calculate range and mid-range levels
  anal->range_ = anal->max_ - anal->min_;
  anal->midRange_ = anal->min_ + anal->range_ / 2.;
  
  // Check if range is above threshold
  if ( anal->range_ < FastFedCablingAnalysis::threshold_ ) {
    anal->addErrorCode(sistrip::smallDataRange_);
    return; 
  }
  
  // Identify samples to be either "low" or "high"
  std::vector<float> high;
  std::vector<float> low;
  for ( uint16_t ibin = 0; ibin < nbins; ibin++ ) { 
    if ( entries[ibin] ) {
      if ( contents[ibin] < anal->midRange_ ) { 
	low.push_back( contents[ibin] ); 
      } else { 
	high.push_back( contents[ibin] ); 
      }
    }
  }

  // Find median of high and low levels
  sort( high.begin(), high.end() );
  sort( low.begin(), low.end() );
  if ( !high.empty() ) { anal->highMedian_ = high[ high.size()%2 ? high.size()/2 : high.size()/2 ]; }
  if ( !low.empty() ) { anal->lowMedian_ = low[ low.size()%2 ? low.size()/2 : low.size()/2 ]; }

  // Check if light levels above thresholds
  //if ( anal->highMedian_ < FastFedCablingAnalysis::dirtyThreshold_ ) { anal->addErrorCode(sistrip::invalidLightLevel_); }
  //if ( anal->lowMedian_ < FastFedCablingAnalysis::trimDacThreshold_ ) { anal->addErrorCode(sistrip::invalidTrimDacLevel_); }
  
  // Find mean and rms in "low" samples
  anal->lowMean_ = 0.;
  anal->lowRms_ = 0.;
  for ( uint16_t ibin = 0; ibin < low.size(); ibin++ ) {
    anal->lowMean_ += low[ibin];
    anal->lowRms_ += low[ibin] * low[ibin];
  }
  if ( !low.empty() ) { 
    anal->lowMean_ = anal->lowMean_ / low.size();
    anal->lowRms_ = anal->lowRms_ / low.size();
  } else { 
    anal->lowMean_ = 1. * sistrip::invalid_;
    anal->lowRms_ = 1. * sistrip::invalid_;
  }
  if ( anal->lowMean_ < sistrip::valid_ ) { 
    anal->lowRms_ = sqrt( fabs(anal->lowRms_-anal->lowMean_*anal->lowMean_) ); 
  } else {
    anal->lowMean_ = 1. * sistrip::invalid_;
    anal->lowRms_ = 1. * sistrip::invalid_;
  }

  // Find mean and rms in "high" samples
  anal->highMean_ = 0.;
  anal->highRms_ = 0.;
  for ( uint16_t ibin = 0; ibin < high.size(); ibin++ ) {
    anal->highMean_ += high[ibin];
    anal->highRms_ += high[ibin] * high[ibin];
  }
  if ( !high.empty() ) { 
    anal->highMean_ = anal->highMean_ / high.size();
    anal->highRms_ = anal->highRms_ / high.size();
  } else { 
    anal->highMean_ = 1. * sistrip::invalid_;
    anal->highRms_ = 1. * sistrip::invalid_;
  }
  if ( anal->highMean_ < sistrip::valid_ ) { 
    anal->highRms_ = sqrt( fabs(anal->highRms_- anal->highMean_*anal->highMean_) ); 
  } else {
    anal->highMean_ = 1. * sistrip::invalid_;
    anal->highRms_ = 1. * sistrip::invalid_;
  }

  // Check if light levels above thresholds
  //if ( anal->highMean_ < FastFedCablingAnalysis::dirtyThreshold_ ) { anal->addErrorCode(sistrip::invalidLightLevel_); }
  //if ( anal->lowMean_ < FastFedCablingAnalysis::trimDacThreshold_ ) { anal->addErrorCode(sistrip::invalidTrimDacLevel_); }

  // Recalculate range
  if ( anal->highMean_ < 1. * sistrip::valid_ &&
       anal->lowMean_  < 1. * sistrip::valid_ ) { 
    anal->range_ = anal->highMean_ - anal->lowMean_;
    anal->midRange_ = anal->lowMean_ + anal->range_ / 2.;
  } else { 
    anal->range_ = 1. * sistrip::invalid_;
    anal->midRange_ = 1. * sistrip::invalid_;
  }
  
  // Check if updated range is valid and above threshold 
  if ( anal->range_ > 1. * sistrip::valid_ ||
       anal->range_ < FastFedCablingAnalysis::threshold_ ) {
    anal->addErrorCode(sistrip::smallDataRange_);
    return; 
  }
  
  // Extract DCU id
  anal->dcuHardId_ = 0;
  for ( uint16_t ibin = 0; ibin < FastFedCablingAnalysis::nBitsForDcuId_; ibin++ ) {
    if ( entries[ibin] ) {
      if ( contents[ibin] > anal->midRange_ ) {
	anal->dcuHardId_ += 0xFFFFFFFF & (1<<ibin);
      }
    }
  }
  if ( !anal->dcuHardId_ ) { anal->dcuHardId_ = sistrip::invalid32_; }

  // Extract DCU id
  anal->lldCh_ = 0;
  for ( uint16_t ibin = 0; ibin < FastFedCablingAnalysis::nBitsForLldCh_; ibin++ ) {
    if ( entries[FastFedCablingAnalysis::nBitsForDcuId_+ibin] ) {
      if ( contents[FastFedCablingAnalysis::nBitsForDcuId_+ibin] > anal->midRange_ ) {
	anal->lldCh_ += ( 0x3 & (1<<ibin) );
      }
    }
  }
  anal->lldCh_++; // starts from 1
  if ( !anal->lldCh_ ) { anal->lldCh_ = sistrip::invalid_; }

}
