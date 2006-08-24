#include "DQM/SiStripCommissioningAnalysis/interface/PedestalsAnalysis.h"
#include "TProfile.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;

// -----------------------------------------------------------------------------
// 
void PedestalsAnalysis::analysis( const TProfiles& profs, 
				  PedestalsAnalysis::Monitorables& mons ) {

  // Checks on whether histos exist
  if ( !profs.peds_ ) {
    cerr << "[" << __PRETTY_FUNCTION__ << "]"
	 << " NULL pointer to 'peds and raw noise' histogram!"
	 << endl;
  }
  if ( !profs.noise_ ) {
    cerr << "[" << __PRETTY_FUNCTION__ << "]"
	 << " NULL pointer to 'residuals and noise' histogram!"
	 << endl;
  }
  
  // Checks on size of histos
  if ( profs.peds_->GetNbinsX() != 256 ) {
    cerr << "[" << __PRETTY_FUNCTION__ << "]"
	 << " Unexpected number of bins for 'peds and raw noise' histogram: "
	 << profs.peds_->GetNbinsX() << endl;
  }
  if ( profs.noise_->GetNbinsX() != 256 ) {
    cerr << "[" << __PRETTY_FUNCTION__ << "]"
	 << " Unexpected number of bins for 'residuals and noise' histogram: "
	 << profs.noise_->GetNbinsX() << endl;
  }

  // Iterate through APVs 
  for ( uint16_t iapv = 0; iapv < 2; iapv++ ) {
    // Used to calc mean and rms for peds and noise
    float p_sum = 0., p_sum2 = 0., p_max = -1025., p_min = 1025.;
    float n_sum = 0., n_sum2 = 0., n_max = -1025., n_min = 1025.;
    // Iterate through strips of APV
    for ( uint16_t istr = 0; istr < 128; istr++ ) {
      static uint16_t strip = iapv*2 + istr;
      // Pedestals 
      if ( profs.peds_ ) {
	if ( profs.peds_->GetBinEntries(strip+1) ) {
	  mons.peds_[iapv][istr] = profs.peds_->GetBinContent(strip+1);
	  p_sum += mons.peds_[iapv][istr];
	  p_sum2 += mons.peds_[iapv][istr] * mons.peds_[iapv][istr];
	  if ( p_max > mons.peds_[iapv][istr] ) { p_max = mons.peds_[iapv][istr]; }
	  if ( p_min < mons.peds_[iapv][istr] ) { p_min = mons.peds_[iapv][istr]; }
	}
      } 
      // Noise
      if ( profs.noise_ ) {
	if ( profs.noise_->GetBinEntries(strip+1) ) {
	  mons.noise_[iapv][istr] = profs.noise_->GetBinError(strip+1);
	  n_sum += mons.noise_[iapv][istr];
	  n_sum2 += mons.noise_[iapv][istr] * mons.noise_[iapv][istr];
	  if ( n_max > mons.noise_[iapv][istr] ) { n_max = mons.noise_[iapv][istr]; }
	  if ( n_min < mons.noise_[iapv][istr] ) { n_min = mons.noise_[iapv][istr]; }
	}
      }

    } // strip loop
    
    // Calc mean and rms for peds
    if ( !mons.peds_[iapv].empty() ) { 
      mons.pedsMean_[iapv] = p_sum / static_cast<float>( mons.peds_[iapv].size() );
      p_sum2 = p_sum2 / static_cast<float>( mons.peds_[iapv].size() );
      if ( p_sum2 > mons.pedsMean_[iapv]*mons.pedsMean_[iapv] ) { 
	mons.pedsSpread_[iapv] = sqrt( p_sum2 - mons.pedsMean_[iapv]*mons.pedsMean_[iapv] );
      }
    }

    // Calc mean and rms for noise
    if ( !mons.noise_[iapv].empty() ) { 
      mons.noiseMean_[iapv] = n_sum / static_cast<float>( mons.noise_[iapv].size() );
      n_sum2 = n_sum2 / static_cast<float>( mons.noise_[iapv].size() );
      if ( n_sum2 > mons.noiseMean_[iapv]*mons.noiseMean_[iapv] ) { 
	mons.noiseSpread_[iapv] = sqrt( n_sum2 - mons.noiseMean_[iapv]*mons.noiseMean_[iapv] );
      }
    }
    
    // Set max and min values for both peds and noise
    if ( p_max > -1024. ) { mons.pedsMax_[iapv] = p_max; }
    if ( p_min < 1024. )  { mons.pedsMin_[iapv] = p_min; }
    if ( n_max > -1024. ) { mons.noiseMax_[iapv] = n_max; }
    if ( n_min < 1024. )  { mons.noiseMin_[iapv] = n_min; }

    // Set dead and noisy strips
    for ( uint16_t istr = 0; istr < 128; istr++ ) {
      if ( mons.noise_[iapv][istr] < (mons.noiseMean_[iapv]-5*mons.noiseSpread_[iapv]) ) {
	mons.dead_[iapv].push_back(istr); //@@ valid threshold???
      } 
      else if ( mons.noise_[iapv][istr] > (mons.noiseMean_[iapv]+5*mons.noiseSpread_[iapv]) ) {
	mons.noisy_[iapv].push_back(istr); //@@ valid threshold???
      }
    }
  
  } // apv loop
  
}

// ----------------------------------------------------------------------------
// 
void PedestalsAnalysis::Monitorables::print( stringstream& ss ) { 
  ss << "FED calibration constants for APV0: " 
     << " Number of pedestal/noise values: " 
     << peds_[0].size() << "/" << noise_[0].size() << "\n"
     << " Number of dead/noisy strips: " 
     << dead_[0].size() << "/" << noisy_[0].size() << "\n"
     << " Mean/Spread/Max/Min pedestal values: "
     << pedsMean_[0] << "/" 
     << pedsSpread_[0] << "/" 
     << pedsMax_[0] << "/" 
     << pedsMin_[0] << "\n"
     << " Mean/Spread/Max/Min noise values: "
     << noiseMean_[0] << "/" 
     << noiseSpread_[0] << "/" 
     << noiseMax_[0] << "/" 
     << noiseMin_[0] << "\n";
  ss << "FED calibration constants for APV0: " 
     << " Number of pedestal/noise values: " 
     << peds_[1].size() << "/" << noise_[1].size() << "\n"
     << " Number of dead/noisy strips: " 
     << dead_[1].size() << "/" << noisy_[1].size() << "\n"
     << " Mean/Spread/Max/Min pedestal values: "
     << pedsMean_[1] << "/" 
     << pedsSpread_[1] << "/" 
     << pedsMax_[1] << "/" 
     << pedsMin_[1] << "\n"
     << " Mean/Spread/Max/Min noise values: "
     << noiseMean_[1] << "/" 
     << noiseSpread_[1] << "/" 
     << noiseMax_[1] << "/" 
     << noiseMin_[1] << "\n";
}

// -----------------------------------------------------------------------------
//
void PedestalsAnalysis::analysis( const vector<const TProfile*>& histos, 
				  vector< vector<float> >& monitorables ) {
  //edm::LogInfo("Commissioning|Analysis") << "[PedestalsAnalysis::analysis]";
  
  if (histos.size() != 2) { 
    // edm::LogError("Commissioning|Analysis") << "[PedestalsAnalysis::analysis]: Requires \"const vector<const TH1F*>& \" argument to have size 2. Actual size: " << histos.size() << ". Monitorables set to 0."; 
    monitorables.push_back(vector<float>(1,0.)); monitorables.push_back(vector<float>(1,0.));
    return; 
  }
  monitorables.resize(2,vector<float>());
  
  // Retrieve histogram contents and set monitorables
  vector<float>& peds = monitorables[0]; 
  vector<float>& noise = monitorables[1]; 
  for ( int ibin = 0; ibin < histos[0]->GetNbinsX(); ibin++ ) {
    peds.push_back( histos[0]->GetBinContent(ibin+1) );
    noise.push_back( histos[1]->GetBinError(ibin+1) );
  }

}
