#include "DQM/SiStripCommissioningAnalysis/interface/PedestalsAnalysis.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TProfile.h"
#include <iomanip>
#include <cmath>

using namespace std;

// -----------------------------------------------------------------------------
// temporarily is wrapping orginal analysis() method
void PedestalsAnalysis::analysis( const vector<const TProfile*>& histos, 
				  PedestalsAnalysis::Monitorables& mons ) {
//   vector<const TProfile*> tmp1; 
//   for ( int ii = 0; ii < histos.size(); ii++ ) { tmp1.push_back(); }
//   vector<unsigned short> tmp2;
//   analysis( tmp1, tmp2 );
//   //@@ to do!
//   for ( uint16_t iapv = 0; iapv = tmp2.size(); iapv++ ) {
//     for ( uint16_t ichan = 0; ichan = tmp2[iapv].size(); ichan++ ) {
//     }
//   }
  
}

// ----------------------------------------------------------------------------
// 
void PedestalsAnalysis::Monitorables::print( stringstream& ss ) { 
  ss << " FED calibration constants for APV0: " 
     << " Number of pedestal/noise values: " 
     << peds_[0].size() << "/" << noise_[0].size() << "\n"
     << " Number of dead/noisy strips: " 
     << dead_[0].size() << "/" << noisy_[0].size() << "\n"
     << " Mean/Spread/Median/Max/Min pedestal values: "
     << setprecision(1) << pedsMean_[0] << "/" 
     << setprecision(1) << pedsSpread_[0] << "/" 
     << setprecision(1) << pedsMedian_[0] << "/" 
     << setprecision(1) << pedsMax_[0] << "/" 
     << setprecision(1) << pedsMin_[0] << "\n"
     << " Mean/Spread/Median/Max/Min noise values: "
     << setprecision(1) << noiseMean_[0] << "/" 
     << setprecision(1) << noiseSpread_[0] << "/" 
     << setprecision(1) << noiseMedian_[0] << "/" 
     << setprecision(1) << noiseMax_[0] << "/" 
     << setprecision(1) << noiseMin_[0] << "\n";
  ss << " FED calibration constants for APV0: " 
     << " Number of pedestal/noise values: " 
     << peds_[1].size() << "/" << noise_[1].size() << "\n"
     << " Number of dead/noisy strips: " 
     << dead_[1].size() << "/" << noisy_[1].size() << "\n"
     << " Mean/Spread/Median/Max/Min pedestal values: "
     << setprecision(1) << pedsMean_[1] << "/" 
     << setprecision(1) << pedsSpread_[1] << "/" 
     << setprecision(1) << pedsMedian_[1] << "/" 
     << setprecision(1) << pedsMax_[1] << "/" 
     << setprecision(1) << pedsMin_[1] << "\n"
     << " Mean/Spread/Median/Max/Min noise values: "
     << setprecision(1) << noiseMean_[1] << "/" 
     << setprecision(1) << noiseSpread_[1] << "/" 
     << setprecision(1) << noiseMedian_[1] << "/" 
     << setprecision(1) << noiseMax_[1] << "/" 
     << setprecision(1) << noiseMin_[1] << "\n";
}

// -----------------------------------------------------------------------------
//
void PedestalsAnalysis::analysis( const vector<const TProfile*>& histos, 
				  vector< vector<float> >& monitorables ) {
  edm::LogInfo("Commissioning|Analysis") << "[PedestalsAnalysis::analysis]";
 
    //check 
  if (histos.size() != 2) { edm::LogError("Commissioning|Analysis") << "[PedestalsAnalysis::analysis]: Requires \"const vector<const TH1F*>& \" argument to have size 2. Actual size: " << histos.size() << ". Monitorables set to 0."; 
  
  monitorables.push_back(vector<float>(1,0.)); monitorables.push_back(vector<float>(1,0.));
  return; }
  
  monitorables.resize(2,vector<float>());

  // Retrieve histogram contents and set monitorables
  vector<float>& peds = monitorables[0]; 
  vector<float>& noise = monitorables[1]; 
  for ( int ibin = 0; ibin < histos[0]->GetNbinsX(); ibin++ ) {
    peds.push_back( histos[0]->GetBinContent(ibin+1) );
    noise.push_back( histos[1]->GetBinError(ibin+1) );
  }
}
