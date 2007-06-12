#include "DQM/SiStripCommissioningAnalysis/interface/PedestalsAnalysis.h"
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
PedestalsAnalysis::PedestalsAnalysis( const uint32_t& key ) 
  : CommissioningAnalysis(key,"PedestalsAnalysis"),
    peds_(2,VFloat(128,sistrip::invalid_)), 
    noise_(2,VFloat(128,sistrip::invalid_)), 
    raw_(2,VFloat(128,sistrip::invalid_)), 
    dead_(2,VInt(0,sistrip::invalid_)), 
    noisy_(2,VInt(0,sistrip::invalid_)),
    pedsMean_(2,sistrip::invalid_), 
    pedsSpread_(2,sistrip::invalid_), 
    noiseMean_(2,sistrip::invalid_), 
    noiseSpread_(2,sistrip::invalid_), 
    rawMean_(2,sistrip::invalid_), 
    rawSpread_(2,sistrip::invalid_), 
    pedsMax_(2,sistrip::invalid_), 
    pedsMin_(2,sistrip::invalid_), 
    noiseMax_(2,sistrip::invalid_), 
    noiseMin_(2,sistrip::invalid_),
    rawMax_(2,sistrip::invalid_), 
    rawMin_(2,sistrip::invalid_),
    hPeds_(0,""),
    hNoise_(0,"")
{
  dead_[0].reserve(256); dead_[1].reserve(256); 
  noisy_[0].reserve(256); noisy_[1].reserve(256);
}

// ----------------------------------------------------------------------------
// 
PedestalsAnalysis::PedestalsAnalysis() 
  : CommissioningAnalysis("PedestalsAnalysis"),
    peds_(2,VFloat(128,sistrip::invalid_)), 
    noise_(2,VFloat(128,sistrip::invalid_)), 
    raw_(2,VFloat(128,sistrip::invalid_)), 
    dead_(2,VInt(0,sistrip::invalid_)), 
    noisy_(2,VInt(0,sistrip::invalid_)),
    pedsMean_(2,sistrip::invalid_), 
    pedsSpread_(2,sistrip::invalid_), 
    noiseMean_(2,sistrip::invalid_), 
    noiseSpread_(2,sistrip::invalid_), 
    rawMean_(2,sistrip::invalid_), 
    rawSpread_(2,sistrip::invalid_), 
    pedsMax_(2,sistrip::invalid_), 
    pedsMin_(2,sistrip::invalid_), 
    noiseMax_(2,sistrip::invalid_), 
    noiseMin_(2,sistrip::invalid_),
    rawMax_(2,sistrip::invalid_), 
    rawMin_(2,sistrip::invalid_),
    hPeds_(0,""),
    hNoise_(0,"")
{
  dead_[0].reserve(256); dead_[1].reserve(256); 
  noisy_[0].reserve(256); noisy_[1].reserve(256);
}

// ----------------------------------------------------------------------------
// 
void PedestalsAnalysis::reset() {
  peds_        = VVFloat(2,VFloat(128,sistrip::invalid_)); 
  noise_       = VVFloat(2,VFloat(128,sistrip::invalid_)); 
  raw_         = VVFloat(2,VFloat(128,sistrip::invalid_));
  dead_        = VVInt(2,VInt(0,sistrip::invalid_)); 
  noisy_       = VVInt(2,VInt(0,sistrip::invalid_));
  pedsMean_    = VFloat(2,sistrip::invalid_); 
  pedsSpread_  = VFloat(2,sistrip::invalid_); 
  noiseMean_   = VFloat(2,sistrip::invalid_); 
  noiseSpread_ = VFloat(2,sistrip::invalid_); 
  rawMean_     = VFloat(2,sistrip::invalid_);
  rawSpread_   = VFloat(2,sistrip::invalid_);
  pedsMax_     = VFloat(2,sistrip::invalid_); 
  pedsMin_     = VFloat(2,sistrip::invalid_); 
  noiseMax_    = VFloat(2,sistrip::invalid_); 
  noiseMin_    = VFloat(2,sistrip::invalid_);
  rawMax_      = VFloat(2,sistrip::invalid_);
  rawMin_      = VFloat(2,sistrip::invalid_);
  dead_[0].reserve(256); 
  dead_[1].reserve(256); 
  noisy_[0].reserve(256); 
  noisy_[1].reserve(256);
  hPeds_ = Histo(0,"");
  hNoise_ = Histo(0,"");
}

// ----------------------------------------------------------------------------
// 
void PedestalsAnalysis::extract( const std::vector<TH1*>& histos ) { 

  // Check
  if ( histos.size() != 2 ) {
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
    
    // Check run type
    SiStripHistoTitle title( (*ihis)->GetName() );
    if ( title.runType() != sistrip::PEDESTALS ) {
      edm::LogWarning(mlCommissioning_) 
	<< "[" << myName() << "::" << __func__ << "]"
	<< " Unexpected commissioning task: "
	<< SiStripEnumsAndStrings::runType(title.runType());
      continue;
    }
    
    // Extract peds and noise histos
    if ( title.extraInfo().find(sistrip::pedsAndRawNoise_) != std::string::npos ) {
      hPeds_.first = *ihis;
      hPeds_.second = (*ihis)->GetName();
    } else if ( title.extraInfo().find(sistrip::residualsAndNoise_) != std::string::npos ) {
      hNoise_.first = *ihis;
      hNoise_.second = (*ihis)->GetName();
    } else if ( title.extraInfo().find(sistrip::commonMode_) != std::string::npos ) {
      //@@ something here for CM plots?
    } else { 
      edm::LogWarning(mlCommissioning_)
	<< "[" << myName() << "::" << __func__ << "]"
	<< " Unexpected 'extra info': " << title.extraInfo();
    }
    
  }

}

// -----------------------------------------------------------------------------
// 
void PedestalsAnalysis::analyse() {

  // Checks on whether pedestals histo exists and if binning is correct
  if ( hPeds_.first ) {
    if ( hPeds_.first->GetNbinsX() != 256 ) {
      edm::LogWarning(mlCommissioning_) 
	<< "[" << myName() << "::" << __func__ << "]"
	<< " Unexpected number of bins for 'peds and raw noise' histogram: "
	<< hPeds_.first->GetNbinsX();
    }
  } else { 
    edm::LogWarning(mlCommissioning_)
      << "[" << myName() << "::" << __func__ << "]"
      << " NULL pointer to 'peds and raw noise' histogram!";
    return;
  }
  
  // Checks on whether noise histo exists and if binning is correct
  if ( hNoise_.first ) {
    if ( hNoise_.first->GetNbinsX() != 256 ) {
      edm::LogWarning(mlCommissioning_) 
	<< "[" << myName() << "::" << __func__ << "]"
	<< " Unexpected number of bins for 'residuals and noise' histogram: "
	<< hNoise_.first->GetNbinsX();
    }
  } else {
    edm::LogWarning(mlCommissioning_)
      << "[" << myName() << "::" << __func__ << "]"
      << " NULL pointer to 'residuals and noise' histogram!";
    return;
  }

  // Extract TProfile histograms
  TProfile* peds_histo = dynamic_cast<TProfile*>(hPeds_.first);
  TProfile* noise_histo = dynamic_cast<TProfile*>(hNoise_.first);

  // Checks on whether pedestals TProfile histo exists
  if ( !peds_histo ) {
    edm::LogWarning(mlCommissioning_) 
      << "[" << myName() << "::" << __func__ << "]"
      << " NULL pointer to 'peds and raw noise' TProfile histogram!";
    return;
  }

  // Checks on whether noise TProfile histo exists
  if ( !noise_histo ) {
    edm::LogWarning(mlCommissioning_) 
      << "[" << myName() << "::" << __func__ << "]"
      << " NULL pointer to 'residuals and noise' TProfile histogram!";
    return;
  }
  
  // Iterate through APVs 
  for ( uint16_t iapv = 0; iapv < 2; iapv++ ) {

    // Used to calc mean and rms for peds and noise
    float p_sum = 0., p_sum2 = 0., p_max = -1.*sistrip::invalid_, p_min = sistrip::invalid_;
    float n_sum = 0., n_sum2 = 0., n_max = -1.*sistrip::invalid_, n_min = sistrip::invalid_;
    float r_sum = 0., r_sum2 = 0., r_max = -1.*sistrip::invalid_, r_min = sistrip::invalid_;

    // Iterate through strips of APV
    for ( uint16_t istr = 0; istr < 128; istr++ ) {

      static uint16_t strip;
      strip = iapv*128 + istr;

      // Pedestals and raw noise
      if ( peds_histo ) {
	if ( peds_histo->GetBinEntries(strip+1) ) {

	  peds_[iapv][istr] = peds_histo->GetBinContent(strip+1);
	  p_sum += peds_[iapv][istr];
	  p_sum2 += (peds_[iapv][istr] * peds_[iapv][istr]);
	  if ( peds_[iapv][istr] > p_max ) { p_max = peds_[iapv][istr]; }
	  if ( peds_[iapv][istr] < p_min ) { p_min = peds_[iapv][istr]; }

	  raw_[iapv][istr] = peds_histo->GetBinError(strip+1);
	  r_sum += raw_[iapv][istr];
	  r_sum2 += (raw_[iapv][istr] * raw_[iapv][istr]);
	  if ( raw_[iapv][istr] > r_max ) { r_max = raw_[iapv][istr]; }
	  if ( raw_[iapv][istr] < r_min ) { r_min = raw_[iapv][istr]; }

	}
      } 

      // Noise
      if ( noise_histo ) {
	if ( noise_histo->GetBinEntries(strip+1) ) {
	  noise_[iapv][istr] = noise_histo->GetBinError(strip+1);
	  n_sum += noise_[iapv][istr];
	  n_sum2 += (noise_[iapv][istr] * noise_[iapv][istr]);
	  if ( noise_[iapv][istr] > n_max ) { n_max = noise_[iapv][istr]; }
	  if ( noise_[iapv][istr] < n_min ) { n_min = noise_[iapv][istr]; }
	}
      }

    } // strip loop
    
    // Calc mean and rms for peds
    if ( !peds_[iapv].empty() ) { 
      p_sum /= static_cast<float>( peds_[iapv].size() );
      p_sum2 /= static_cast<float>( peds_[iapv].size() );
      pedsMean_[iapv] = p_sum;
      pedsSpread_[iapv] = sqrt( fabs(p_sum2 - p_sum*p_sum) );
    }
    
    // Calc mean and rms for noise
    if ( !noise_[iapv].empty() ) { 
      n_sum /= static_cast<float>( noise_[iapv].size() );
      n_sum2 /= static_cast<float>( noise_[iapv].size() );
      noiseMean_[iapv] = n_sum;
      noiseSpread_[iapv] = sqrt( fabs(n_sum2 - n_sum*n_sum) );
    }

    // Calc mean and rms for raw noise
    if ( !raw_[iapv].empty() ) { 
      r_sum /= static_cast<float>( raw_[iapv].size() );
      r_sum2 /= static_cast<float>( raw_[iapv].size() );
      rawMean_[iapv] = r_sum;
      rawSpread_[iapv] = sqrt( fabs(r_sum2 - r_sum*r_sum) );
    }
    
    // Set max and min values for peds, noise and raw noise
    if ( p_max > -1.*sistrip::maximum_ ) { pedsMax_[iapv] = p_max; }
    if ( p_min < 1.*sistrip::maximum_ )  { pedsMin_[iapv] = p_min; }
    if ( n_max > -1.*sistrip::maximum_ ) { noiseMax_[iapv] = n_max; }
    if ( n_min < 1.*sistrip::maximum_ )  { noiseMin_[iapv] = n_min; }
    if ( r_max > -1.*sistrip::maximum_ ) { rawMax_[iapv] = r_max; }
    if ( r_min < 1.*sistrip::maximum_ )  { rawMin_[iapv] = r_min; }
    
    // Set dead and noisy strips
    for ( uint16_t istr = 0; istr < 128; istr++ ) {
      if ( noiseMin_[iapv] > sistrip::maximum_ ||
	   noiseMax_[iapv] > sistrip::maximum_ ) { continue; }
      if ( noise_[iapv][istr] < (noiseMean_[iapv] - 5.*noiseSpread_[iapv]) ) {
	dead_[iapv].push_back(istr); //@@ valid threshold???
      } 
      else if ( noise_[iapv][istr] > (noiseMean_[iapv] + 5.*noiseSpread_[iapv]) ) {
	noisy_[iapv].push_back(istr); //@@ valid threshold???
      }
    }
    
  } // apv loop

}

// ----------------------------------------------------------------------------
// 
bool PedestalsAnalysis::isValid() const {
  return ( pedsMean_[0] < sistrip::maximum_ &&
	   pedsMean_[1] < sistrip::maximum_ &&
	   pedsSpread_[0] < sistrip::maximum_ &&
	   pedsSpread_[1] < sistrip::maximum_ &&
	   noiseMean_[0] < sistrip::maximum_ &&
	   noiseMean_[1] < sistrip::maximum_ &&
	   noiseSpread_[0] < sistrip::maximum_ &&
	   noiseSpread_[1] < sistrip::maximum_ &&
	   rawMean_[0] < sistrip::maximum_ &&
	   rawMean_[1] < sistrip::maximum_ &&
	   rawSpread_[0] < sistrip::maximum_ &&
	   rawSpread_[1] < sistrip::maximum_ &&
	   pedsMax_[0] < sistrip::maximum_ &&
	   pedsMax_[1] < sistrip::maximum_ &&
	   pedsMin_[0] < sistrip::maximum_ &&
	   pedsMin_[1] < sistrip::maximum_ &&
	   noiseMax_[0] < sistrip::maximum_ &&
	   noiseMax_[1] < sistrip::maximum_ &&
	   noiseMin_[0] < sistrip::maximum_ &&
	   noiseMin_[1] < sistrip::maximum_ &&
	   rawMax_[0] < sistrip::maximum_ &&
	   rawMax_[1] < sistrip::maximum_ &&
	   rawMin_[0] < sistrip::maximum_ &&
	   rawMin_[1] < sistrip::maximum_ &&
	   noiseMean_[0] <= rawMean_[0] && //@@ temp
	   noiseMean_[1] <= rawMean_[1] ); //@@ temp
} 

// ----------------------------------------------------------------------------
// 
void PedestalsAnalysis::print( std::stringstream& ss, uint32_t iapv ) { 
  if ( iapv == 1 || iapv == 2 ) { iapv--; }
  else { iapv = 0; }
  header( ss );
  ss << " Monitorables for APV number     : " << iapv;
  if ( iapv == 0 ) { ss << " (first of pair)"; }
  else if ( iapv == 1 ) { ss << " (second of pair)"; } 
  ss << std::endl;
  ss << " Number of pedestal values       : " << peds_[iapv].size() << std::endl
     << " Number of noise values          : " << noise_[iapv].size() << std::endl
     << " Number of raw noise values      : " << raw_[iapv].size() << std::endl
     << " Dead strips  (<5s) [strip]      : (" << dead_[iapv].size() << " in total) ";
  for ( uint16_t ii = 0; ii < dead_[iapv].size(); ii++ ) { 
    ss << dead_[iapv][ii] << " "; }
  
  ss << std::endl;
  ss << " Noisy strips (>5s) [strip]      : (" << noisy_[iapv].size() << " in total) ";
  for ( uint16_t ii = 0; ii < noisy_[iapv].size(); ii++ ) { 
    ss << noisy_[iapv][ii] << " "; 
  } 
  ss << std::endl;
  ss << std::fixed << std::setprecision(2)
     << " Mean peds +/- spread [adc]      : " << pedsMean_[iapv] << " +/- " << pedsSpread_[iapv] << std::endl 
     << " Min/Max pedestal [adc]          : " << pedsMin_[iapv] << " <-> " << pedsMax_[iapv] << std::endl
     << " Mean noise +/- spread [adc]     : " << noiseMean_[iapv] << " +/- " << noiseSpread_[iapv] << std::endl 
     << " Min/Max noise [adc]             : " << noiseMin_[iapv] << " <-> " << noiseMax_[iapv] << std::endl
     << " Mean raw noise +/- spread [adc] : " << rawMean_[iapv] << " +/- " << rawSpread_[iapv] << std::endl 
     << " Min/Max raw noise [adc]         : " << rawMin_[iapv] << " <-> " << rawMax_[iapv] << std::endl
     << " Normalised noise                : " << "(yet to be implemented...)" << std::endl
     << std::boolalpha 
     << " isValid                         : " << isValid()  << std::endl
     << std::noboolalpha;
}

