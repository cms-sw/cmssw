#include "DQM/SiStripCommissioningAnalysis/interface/PedsOnlyAlgorithm.h"
#include "CondFormats/SiStripObjects/interface/PedsOnlyAnalysis.h" 
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
PedsOnlyAlgorithm::PedsOnlyAlgorithm( const edm::ParameterSet & pset, PedsOnlyAnalysis* const anal ) 
  : CommissioningAlgorithm(anal),
    hPeds_(0,""),
    hNoise_(0,"")
{}

// ----------------------------------------------------------------------------
// 
void PedsOnlyAlgorithm::extract( const std::vector<TH1*>& histos ) { 

  if ( !anal() ) {
    edm::LogWarning(mlCommissioning_)
      << "[PedsOnlyAlgorithm::" << __func__ << "]"
      << " NULL pointer to Analysis object!";
    return; 
  }

  // Check number of histograms
  if ( histos.size() != 2 ) {
    anal()->addErrorCode(sistrip::numberOfHistos_);
  }
  
  // Extract FED key from histo title
  if ( !histos.empty() ) { anal()->fedKey( extractFedKey( histos.front() ) ); }
  
  // Extract histograms
  std::vector<TH1*>::const_iterator ihis = histos.begin();
  for ( ; ihis != histos.end(); ihis++ ) {
    
    // Check for NULL pointer
    if ( !(*ihis) ) { continue; }

    // Check run type
    SiStripHistoTitle title( (*ihis)->GetName() );
    if ( title.runType() != sistrip::PEDS_ONLY ) {
      anal()->addErrorCode(sistrip::unexpectedTask_);
      continue;
    }
    
    // Extract peds and raw noise histos (check for legacy names first!)
    if ( title.extraInfo().find(sistrip::extrainfo::pedsAndRawNoise_) != std::string::npos ) {
      hPeds_.first = *ihis;
      hPeds_.second = (*ihis)->GetName();
      hNoise_.first = *ihis;
      hNoise_.second = (*ihis)->GetName();
      PedsOnlyAnalysis* a = dynamic_cast<PedsOnlyAnalysis*>( const_cast<CommissioningAnalysis*>( anal() ) );
      if ( a ) { a->legacy_ = true; }
    } else if ( title.extraInfo().find(sistrip::extrainfo::pedestals_) != std::string::npos ) {
      hPeds_.first = *ihis;
      hPeds_.second = (*ihis)->GetName();
    } else if ( title.extraInfo().find(sistrip::extrainfo::rawNoise_) != std::string::npos ) {
      hNoise_.first = *ihis;
      hNoise_.second = (*ihis)->GetName();
    } else { 
      anal()->addErrorCode(sistrip::unexpectedExtraInfo_);
    }
    
  }

}

// -----------------------------------------------------------------------------
// 
void PedsOnlyAlgorithm::analyse() {
  
  if ( !anal() ) {
    edm::LogWarning(mlCommissioning_)
      << "[PedsOnlyAlgorithm::" << __func__ << "]"
      << " NULL pointer to base Analysis object!";
    return; 
  }
  
  CommissioningAnalysis* tmp = const_cast<CommissioningAnalysis*>( anal() );
  PedsOnlyAnalysis* anal = dynamic_cast<PedsOnlyAnalysis*>( tmp );
  if ( !anal ) {
    edm::LogWarning(mlCommissioning_)
      << "[PedsOnlyAlgorithm::" << __func__ << "]"
      << " NULL pointer to derived Analysis object!";
    return; 
  }
  
  if ( !hPeds_.first ) {
    anal->addErrorCode(sistrip::nullPtr_);
    return;
  }
  
  if ( !hNoise_.first ) {
    anal->addErrorCode(sistrip::nullPtr_);
    return;
  }
  
  TProfile* peds_histo = dynamic_cast<TProfile*>(hPeds_.first);
  TProfile* raw_histo = dynamic_cast<TProfile*>(hNoise_.first);
  
  if ( !peds_histo ) {
    anal->addErrorCode(sistrip::nullPtr_);
    return;
  }

  if ( !raw_histo ) {
    anal->addErrorCode(sistrip::nullPtr_);
    return;
  }

  if ( peds_histo->GetNbinsX() != 256 ) {
    anal->addErrorCode(sistrip::numberOfBins_);
    return;
  }

  if ( raw_histo->GetNbinsX() != 256 ) {
    anal->addErrorCode(sistrip::numberOfBins_);
    return;
  }
  
  // Iterate through APVs 
  for ( uint16_t iapv = 0; iapv < 2; iapv++ ) {

    // Used to calc mean and rms for peds and noise
    float p_sum = 0., p_sum2 = 0., p_max = -1.*sistrip::invalid_, p_min = sistrip::invalid_;
    float r_sum = 0., r_sum2 = 0., r_max = -1.*sistrip::invalid_, r_min = sistrip::invalid_;

    // Iterate through strips of APV
    for ( uint16_t istr = 0; istr < 128; istr++ ) {

      static uint16_t strip;
      strip = iapv*128 + istr;

      // Pedestals 
      if ( peds_histo ) {
	if ( peds_histo->GetBinEntries(strip+1) ) {
	  anal->peds_[iapv][istr] = peds_histo->GetBinContent(strip+1);
	  p_sum += anal->peds_[iapv][istr];
	  p_sum2 += (anal->peds_[iapv][istr] * anal->peds_[iapv][istr]);
	  if ( anal->peds_[iapv][istr] > p_max ) { p_max = anal->peds_[iapv][istr]; }
	  if ( anal->peds_[iapv][istr] < p_min ) { p_min = anal->peds_[iapv][istr]; }
	}
      } 
      
      // Raw noise
      if ( !anal->legacy_ ) {
	if ( raw_histo ) {
	  if ( raw_histo->GetBinEntries(strip+1) ) {
	    anal->raw_[iapv][istr] = raw_histo->GetBinContent(strip+1);
	    r_sum += anal->raw_[iapv][istr];
	    r_sum2 += ( anal->raw_[iapv][istr] * anal->raw_[iapv][istr] );
	    if ( anal->raw_[iapv][istr] > r_max ) { r_max = anal->raw_[iapv][istr]; }
	    if ( anal->raw_[iapv][istr] < r_min ) { r_min = anal->raw_[iapv][istr]; }
	  }
	}
      } else {
	if ( peds_histo ) {
	  if ( peds_histo->GetBinEntries(strip+1) ) {
	    anal->raw_[iapv][istr] = raw_histo->GetBinError(strip+1);
	    r_sum += anal->raw_[iapv][istr];
	    r_sum2 += ( anal->raw_[iapv][istr] * anal->raw_[iapv][istr] );
	    if ( anal->raw_[iapv][istr] > r_max ) { r_max = anal->raw_[iapv][istr]; }
	    if ( anal->raw_[iapv][istr] < r_min ) { r_min = anal->raw_[iapv][istr]; }
	  }
	}
      }
      
    } // strip loop
    
    // Calc mean and rms for peds
    if ( !anal->peds_[iapv].empty() ) { 
      p_sum /= static_cast<float>( anal->peds_[iapv].size() );
      p_sum2 /= static_cast<float>( anal->peds_[iapv].size() );
      anal->pedsMean_[iapv] = p_sum;
      anal->pedsSpread_[iapv] = sqrt( fabs(p_sum2 - p_sum*p_sum) );
    }
    
    // Calc mean and rms for raw noise
    if ( !anal->raw_[iapv].empty() ) { 
      r_sum /= static_cast<float>( anal->raw_[iapv].size() );
      r_sum2 /= static_cast<float>( anal->raw_[iapv].size() );
      anal->rawMean_[iapv] = r_sum;
      anal->rawSpread_[iapv] = sqrt( fabs(r_sum2 - r_sum*r_sum) );
    }
    
    // Set max and min values for peds and raw noise
    if ( p_max > -1.*sistrip::maximum_ ) { anal->pedsMax_[iapv] = p_max; }
    if ( p_min < 1.*sistrip::maximum_ )  { anal->pedsMin_[iapv] = p_min; }
    if ( r_max > -1.*sistrip::maximum_ ) { anal->rawMax_[iapv] = r_max; }
    if ( r_min < 1.*sistrip::maximum_ )  { anal->rawMin_[iapv] = r_min; }
    
  } // apv loop

}
