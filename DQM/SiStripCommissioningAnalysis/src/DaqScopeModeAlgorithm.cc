#include "DQM/SiStripCommissioningAnalysis/interface/DaqScopeModeAlgorithm.h"
#include "CondFormats/SiStripObjects/interface/DaqScopeModeAnalysis.h" 
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TH1.h"
#include "TProfile.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace sistrip;

// ----------------------------------------------------------------------------
// 
DaqScopeModeAlgorithm::DaqScopeModeAlgorithm( const edm::ParameterSet & pset, DaqScopeModeAnalysis* const anal ) 
  : CommissioningAlgorithm(anal),
    histo_(nullptr,""),
    hPeds_(nullptr,""),
    hNoise_(nullptr,""),
    deadStripMax_(pset.getParameter<double>("DeadStripMax")),
    noisyStripMin_(pset.getParameter<double>("NoisyStripMin"))
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
  if ( histos.size() != 3 ) {
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
    // Extract peds and noise histos (check for legacy names first!)
    if ( title.extraInfo().find(sistrip::extrainfo::pedsAndRawNoise_) != std::string::npos ) {
      hPeds_.first = *ihis;
      hPeds_.second = (*ihis)->GetName();
    } 
    else if ( title.extraInfo().find(sistrip::extrainfo::pedsAndCmSubNoise_) != std::string::npos ) {
      hNoise_.first = *ihis;
      hNoise_.second = (*ihis)->GetName();
    } else if ( title.extraInfo().find(sistrip::extrainfo::pedestals_) != std::string::npos ) {
      hPeds_.first = *ihis;
      hPeds_.second = (*ihis)->GetName();
    } else if ( title.extraInfo().find(sistrip::extrainfo::noise_) != std::string::npos ) {
      hNoise_.first = *ihis;
      hNoise_.second = (*ihis)->GetName();
    } else if ( title.extraInfo().find(sistrip::extrainfo::commonMode_) != std::string::npos ) {
      //@@ something here for CM plots?
    }
    else if ( title.extraInfo().find(sistrip::extrainfo::scopeModeFrame_) != std::string::npos ) {
      histo_.first = *ihis;
      histo_.second = (*ihis)->GetName();
    }
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

  // Analysis level wants all the informations --> it will work only on Spy-events
  if ( !hPeds_.first ) {
    anal->addErrorCode(sistrip::nullPtr_);
    return;
  }

  if ( !hNoise_.first ) {
    anal->addErrorCode(sistrip::nullPtr_);
    return;
  }

  if ( !histo_.first ) {
    anal->addErrorCode(sistrip::nullPtr_);
    return;
  }

  /// pedestal
  TProfile* peds_histo  = dynamic_cast<TProfile*>(hPeds_.first);
  TProfile* noise_histo = dynamic_cast<TProfile*>(hNoise_.first);
  /// scope-mode profile
  TProfile* scope_histo = dynamic_cast<TProfile*>(histo_.first);

  if ( !peds_histo ) {
    anal->addErrorCode(sistrip::nullPtr_);
    return;
  }

  if ( !noise_histo ) {
    anal->addErrorCode(sistrip::nullPtr_);
    return;
  }

  if ( !scope_histo ) {
    anal->addErrorCode(sistrip::nullPtr_);
    return;
  }

  if ( peds_histo->GetNbinsX() != 256 ) {
    anal->addErrorCode(sistrip::numberOfBins_);
    return;
  }

  if ( noise_histo->GetNbinsX() != 256 ) {
    anal->addErrorCode(sistrip::numberOfBins_);
    return;
  }

  if ( scope_histo->GetNbinsX() != 298 ) {
    anal->addErrorCode(sistrip::numberOfBins_);
    return;
  }

  // Calculate pedestals and noise
  // Iterate through APVs 
  for ( uint16_t iapv = 0; iapv < 2; iapv++ ) {
    
    // Used to calc mean and rms for peds and noise
    float p_sum = 0., p_sum2 = 0., p_max = -1.*sistrip::invalid_, p_min = sistrip::invalid_;
    float n_sum = 0., n_sum2 = 0., n_max = -1.*sistrip::invalid_, n_min = sistrip::invalid_;
    float r_sum = 0., r_sum2 = 0., r_max = -1.*sistrip::invalid_, r_min = sistrip::invalid_;
    
    // Iterate through strips of APV
    for ( uint16_t istr = 0; istr < 128; istr++ ) {      
      uint16_t strip = iapv*128 + istr;
      
      // Pedestals and raw noise
      if ( peds_histo ) {
	if ( peds_histo->GetBinEntries(strip+1) ) {
	  anal->peds_[iapv][istr] = peds_histo->GetBinContent(strip+1);
	  p_sum  += anal->peds_[iapv][istr];
	  p_sum2 += (anal->peds_[iapv][istr] * anal->peds_[iapv][istr]);
	  if ( anal->peds_[iapv][istr] > p_max ) { p_max = anal->peds_[iapv][istr]; }
	  if ( anal->peds_[iapv][istr] < p_min ) { p_min = anal->peds_[iapv][istr]; }

	  anal->raw_[iapv][istr] = peds_histo->GetBinError(strip+1);
	  r_sum  += anal->raw_[iapv][istr];
	  r_sum2 += (anal->raw_[iapv][istr] * anal->raw_[iapv][istr]);
	  if ( anal->raw_[iapv][istr] > r_max ) { r_max = anal->raw_[iapv][istr]; }
	  if ( anal->raw_[iapv][istr] < r_min ) { r_min = anal->raw_[iapv][istr]; }
	}
      }
      
      // Noise
      if ( noise_histo ) {
	if ( noise_histo->GetBinEntries(strip+1) ) {
	  anal->noise_[iapv][istr] = noise_histo->GetBinContent(strip+1);
	  n_sum  += anal->noise_[iapv][istr];
	  n_sum2 += (anal->noise_[iapv][istr] * anal->noise_[iapv][istr]);
	  if ( anal->noise_[iapv][istr] > n_max ) { n_max = anal->noise_[iapv][istr]; }
	  if ( anal->noise_[iapv][istr] < n_min ) { n_min = anal->noise_[iapv][istr]; }
	}
      }      
    } // strip loop
    
    // Calc mean and rms for peds
    if (!anal->peds_[iapv].empty()) { 
      p_sum  /= static_cast<float>( anal->peds_[iapv].size() );
      p_sum2 /= static_cast<float>( anal->peds_[iapv].size() );
      anal->pedsMean_[iapv] = p_sum;
      anal->pedsSpread_[iapv] = sqrt( fabs(p_sum2 - p_sum*p_sum) );
    }
    
    // Calc mean and rms for noise
    if ( !anal->noise_[iapv].empty() ) { 
      n_sum /= static_cast<float>( anal->noise_[iapv].size() );
      n_sum2 /= static_cast<float>( anal->noise_[iapv].size() );
      anal->noiseMean_[iapv] = n_sum;
      anal->noiseSpread_[iapv] = sqrt( fabs(n_sum2 - n_sum*n_sum) );
    }

    // Calc mean and rms for raw noise
    if ( !anal->raw_[iapv].empty() ) { 
      r_sum /= static_cast<float>( anal->raw_[iapv].size() );
      r_sum2 /= static_cast<float>( anal->raw_[iapv].size() );
      anal->rawMean_[iapv] = r_sum;
      anal->rawSpread_[iapv] = sqrt( fabs(r_sum2 - r_sum*r_sum) );
    }

    // Set max and min values for peds, noise and raw noise
    if ( p_max > -1.*sistrip::maximum_ ) { anal->pedsMax_[iapv] = p_max; }
    if ( p_min < 1.*sistrip::maximum_ )  { anal->pedsMin_[iapv] = p_min; }
    if ( n_max > -1.*sistrip::maximum_ ) { anal->noiseMax_[iapv] = n_max; }
    if ( n_min < 1.*sistrip::maximum_ )  { anal->noiseMin_[iapv] = n_min; }
    if ( r_max > -1.*sistrip::maximum_ ) { anal->rawMax_[iapv] = r_max; }
    if ( r_min < 1.*sistrip::maximum_ )  { anal->rawMin_[iapv] = r_min; }

    // Set dead and noisy strips
    for ( uint16_t istr = 0; istr < 128; istr++ ) {
      if ( anal->noiseMin_[iapv] > sistrip::maximum_ ||
	   anal->noiseMax_[iapv] > sistrip::maximum_ ) { continue; }
      if ( anal->noise_[iapv][istr] < ( anal->noiseMean_[iapv] - deadStripMax_ * anal->noiseSpread_[iapv] ) ) {
	anal->dead_[iapv].push_back(istr);
      } 
      else if ( anal->noise_[iapv][istr] > ( anal->noiseMean_[iapv] + noisyStripMin_ * anal->noiseSpread_[iapv] ) ) {
	anal->noisy_[iapv].push_back(istr);
      }
    }   
  } // apv loop

  //// Tick-Mark --> just store values and check if the height is significant
  anal->peak_ = (scope_histo->GetBinContent(287)+scope_histo->GetBinContent(288))/2.; // trailing tickmark for each APV
  anal->base_ = (scope_histo->GetBinContent(1)+scope_histo->GetBinContent(2)+scope_histo->GetBinContent(3)+scope_histo->GetBinContent(4)+scope_histo->GetBinContent(5))/5.;
  anal->height_ = anal->peak_-anal->base_;
  if ( anal->height_ < DaqScopeModeAnalysis::tickMarkHeightThreshold_ ) {
    anal->addErrorCode(sistrip::smallTickMarkHeight_);
    return; 
  }    
}
