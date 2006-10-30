#include "DQM/SiStripCommissioningAnalysis/interface/PedestalsAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "TProfile.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;

// ----------------------------------------------------------------------------
// 
PedestalsAnalysis::PedestalsAnalysis( const uint32_t& key ) 
  : CommissioningAnalysis(key),
    peds_(2,VFloats(128,sistrip::invalid_)), 
    noise_(2,VFloats(128,sistrip::invalid_)), 
    dead_(2,VInts(0,sistrip::invalid_)), 
    noisy_(2,VInts(0,sistrip::invalid_)),
    pedsMean_(2,sistrip::invalid_), 
    pedsSpread_(2,sistrip::invalid_), 
    noiseMean_(2,sistrip::invalid_), 
    noiseSpread_(2,sistrip::invalid_), 
    pedsMax_(2,sistrip::invalid_), 
    pedsMin_(2,sistrip::invalid_), 
    noiseMax_(2,sistrip::invalid_), 
    noiseMin_(2,sistrip::invalid_),
    hPeds_(0,""),
    hNoise_(0,"")
{
  dead_[0].reserve(256); dead_[1].reserve(256); 
  noisy_[0].reserve(256); noisy_[1].reserve(256);
}

// ----------------------------------------------------------------------------
// 
PedestalsAnalysis::PedestalsAnalysis() 
  : CommissioningAnalysis(),
    peds_(2,VFloats(128,sistrip::invalid_)), 
    noise_(2,VFloats(128,sistrip::invalid_)), 
    dead_(2,VInts(0,sistrip::invalid_)), 
    noisy_(2,VInts(0,sistrip::invalid_)),
    pedsMean_(2,sistrip::invalid_), 
    pedsSpread_(2,sistrip::invalid_), 
    noiseMean_(2,sistrip::invalid_), 
    noiseSpread_(2,sistrip::invalid_), 
    pedsMax_(2,sistrip::invalid_), 
    pedsMin_(2,sistrip::invalid_), 
    noiseMax_(2,sistrip::invalid_), 
    noiseMin_(2,sistrip::invalid_),
    hPeds_(0,""),
    hNoise_(0,"")
{
  dead_[0].reserve(256); dead_[1].reserve(256); 
  noisy_[0].reserve(256); noisy_[1].reserve(256);
}

// ----------------------------------------------------------------------------
// 
void PedestalsAnalysis::print( stringstream& ss, uint32_t iapv ) { 
  if ( iapv != 0 && iapv != 1 ) { iapv = 0; }
  
  if ( key() ) {
    ss << "FED calibration constants for channel key 0x"
       << hex << setw(8) << setfill('0') << key() << dec 
       << " and APV" << iapv << "\n";
  } else {
    ss << "FED calibration constants for APV" << iapv << "\n";
  }
  ss << " Number of pedestal values   : " << peds_[iapv].size() << "\n"
     << " Number of noise values      : " << noise_[iapv].size() << "\n"
     << " Dead strips  (>5s) [strip]  : (" << dead_[iapv].size() << " in total) ";
  for ( uint16_t ii = 0; ii < dead_[iapv].size(); ii++ ) { 
    ss << dead_[iapv][ii] << " "; }
  
  ss << "\n";
  ss << " Noisy strips (<5s) [strip]  : (" << noisy_[iapv].size() << " in total) ";
  for ( uint16_t ii = 0; ii < noisy_[iapv].size(); ii++ ) { 
    ss << noisy_[iapv][ii] << " "; 
  } 
  ss << "\n";
  ss << " Mean peds +/- spread [adc]  : " << pedsMean_[iapv] << " +/- " << pedsSpread_[iapv] << "\n" 
     << " Max/Min pedestal [adc]      : " << pedsMax_[iapv] << " <-> " << pedsMin_[iapv] << "\n"
     << " Mean noise +/- spread [adc] : " << noiseMean_[iapv] << " +/- " << noiseSpread_[iapv] << "\n" 
     << " Max/Min noise [adc]         : " << noiseMax_[iapv] << " <-> " << noiseMin_[iapv] << "\n"
     << " Normalised noise (to come!) : " << "\n";
}

// ----------------------------------------------------------------------------
// 
void PedestalsAnalysis::reset() {
  peds_        = VVFloats(2,VFloats(128,sistrip::invalid_)); 
  noise_       = VVFloats(2,VFloats(128,sistrip::invalid_)); 
  dead_        = VVInts(2,VInts(0,sistrip::invalid_)); 
  noisy_       = VVInts(2,VInts(0,sistrip::invalid_));
  pedsMean_    = VFloats(2,sistrip::invalid_); 
  pedsSpread_  = VFloats(2,sistrip::invalid_); 
  noiseMean_   = VFloats(2,sistrip::invalid_); 
  noiseSpread_ = VFloats(2,sistrip::invalid_); 
  pedsMax_     = VFloats(2,sistrip::invalid_); 
  pedsMin_     = VFloats(2,sistrip::invalid_); 
  noiseMax_    = VFloats(2,sistrip::invalid_); 
  noiseMin_    = VFloats(2,sistrip::invalid_);
  dead_[0].reserve(256); 
  dead_[1].reserve(256); 
  noisy_[0].reserve(256); 
  noisy_[1].reserve(256);
  hPeds_ = Histo(0,"");
  hNoise_ = Histo(0,"");
}

// ----------------------------------------------------------------------------
// 
void PedestalsAnalysis::extract( const vector<TProfile*>& histos ) { 

  // Check
  if ( histos.size() != 2 ) {
    cerr << "[" << __PRETTY_FUNCTION__ << "]"
	 << " Unexpected number of histograms: " 
	 << histos.size()
	 << endl;
  }
  
  // Extract
  vector<TProfile*>::const_iterator ihis = histos.begin();
  for ( ; ihis != histos.end(); ihis++ ) {
    
    // Check pointer
    //cout << "[" << __PRETTY_FUNCTION__ << "] ptr: " << *ihis << endl;
    if ( !(*ihis) ) {
      cerr << "[" << __PRETTY_FUNCTION__ << "]"
	   << " NULL pointer to histogram!" << endl;
      continue;
    }
    
    // Check name
    static HistoTitle title;
    title = SiStripHistoNamingScheme::histoTitle( (*ihis)->GetName() );
    if ( title.task_ != sistrip::PEDESTALS ) {
      cerr << "[" << __PRETTY_FUNCTION__ << "]"
	   << " Unexpected commissioning task!"
	   << "(" << SiStripHistoNamingScheme::task( title.task_ ) << ")"
	   << endl;
      continue;
    }
    
    // Extract peds and noise histos
    if ( title.extraInfo_.find(sistrip::pedsAndRawNoise_) != string::npos ) {
      hPeds_.first = *ihis;
      hPeds_.second = (*ihis)->GetName();
      //cout << "pedsAndRawNoise name: " << hPeds_.second << endl;
    } else if ( title.extraInfo_.find(sistrip::residualsAndNoise_) != string::npos ) {
      hNoise_.first = *ihis;
      hNoise_.second = (*ihis)->GetName();
      //cout << "residualsAndNoise name: " << hPeds_.second << endl;
    } else { 
      cerr << "[" << __PRETTY_FUNCTION__ << "]"
	   << " Unexpected 'extra info': " << title.extraInfo_ << endl;
    }
    
  }

}

// -----------------------------------------------------------------------------
// 
void PedestalsAnalysis::analyse() {

  // Checks on whether pedestals histo exists and if binning is correct
  if ( hPeds_.first ) {
    if ( hPeds_.first->GetNbinsX() != 256 ) {
      cerr << "[" << __PRETTY_FUNCTION__ << "]"
	   << " Unexpected number of bins for 'peds and raw noise' histogram: "
	   << hPeds_.first->GetNbinsX() << endl;
    }
  } else { 
    cerr << "[" << __PRETTY_FUNCTION__ << "]"
	 << " NULL pointer to 'peds and raw noise' histogram!"
	 << endl;
  }
  
  // Checks on whether noise histo exists and if binning is correct
  if ( hNoise_.first ) {
    if ( hNoise_.first->GetNbinsX() != 256 ) {
      cerr << "[" << __PRETTY_FUNCTION__ << "]"
	   << " Unexpected number of bins for 'residuals and noise' histogram: "
	   << hNoise_.first->GetNbinsX() << endl;
    }
  } else {
    cerr << "[" << __PRETTY_FUNCTION__ << "]"
	 << " NULL pointer to 'residuals and noise' histogram!"
	 << endl;
  }
  
  // Iterate through APVs 
  for ( uint16_t iapv = 0; iapv < 2; iapv++ ) {
    // Used to calc mean and rms for peds and noise
    float p_sum = 0., p_sum2 = 0., p_max = -1.*sistrip::invalid_, p_min = sistrip::invalid_;
    float n_sum = 0., n_sum2 = 0., n_max = -1.*sistrip::invalid_, n_min = sistrip::invalid_;
    // Iterate through strips of APV
    for ( uint16_t istr = 0; istr < 128; istr++ ) {
      static uint16_t strip;
      strip = iapv*128 + istr;
      // Pedestals 
      if ( hPeds_.first ) {
	if ( hPeds_.first->GetBinEntries(strip+1) ) {
	  peds_[iapv][istr] = hPeds_.first->GetBinContent(strip+1);
	  p_sum += peds_[iapv][istr];
	  p_sum2 += (peds_[iapv][istr] * peds_[iapv][istr]);
	  if ( peds_[iapv][istr] > p_max ) { p_max = peds_[iapv][istr]; }
	  if ( peds_[iapv][istr] < p_min ) { p_min = peds_[iapv][istr]; }
	}
      } 
      // Noise
      if ( hNoise_.first ) {
	if ( hNoise_.first->GetBinEntries(strip+1) ) {
	  noise_[iapv][istr] = hNoise_.first->GetBinError(strip+1);
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
    
    // Set max and min values for both peds and noise
    if ( p_max > -1024. ) { pedsMax_[iapv] = p_max; }
    if ( p_min < 1024. )  { pedsMin_[iapv] = p_min; }
    if ( n_max > -1024. ) { noiseMax_[iapv] = n_max; }
    if ( n_min < 1024. )  { noiseMin_[iapv] = n_min; }

    // Set dead and noisy strips
    for ( uint16_t istr = 0; istr < 128; istr++ ) {
      if ( noise_[iapv][istr] < (noiseMean_[iapv] - 5.*noiseSpread_[iapv]) ) {
	dead_[iapv].push_back(istr); //@@ valid threshold???
      } 
      else if ( noise_[iapv][istr] > (noiseMean_[iapv] + 5.*noiseSpread_[iapv]) ) {
	noisy_[iapv].push_back(istr); //@@ valid threshold???
      }
    }
  
  } // apv loop
  
}

