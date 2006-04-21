#include "DQM/SiStripCommissioningAnalysis/interface/PedestalsAnalysis.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TH1F.h"
#include <vector>
#include <cmath>

// -----------------------------------------------------------------------------
//
void PedestalsAnalysis::analysis( const vector<const TH1F*>& histos, 
			      vector< vector<float> >& monitorables ) {
  edm::LogInfo("Commissioning|Analysis") << "[PedestalsAnalysis::analysis]";
 
    //check 
  if (histos.size() != 1) { edm::LogError("Commissioning|Analysis") << "[PedestalsAnalysis::analysis]: Requires \"const vector<const TH1F*>& \" argument to have size 1. Actual size: " << histos.size() << ". Monitorables set to 0."; 
  
  monitorables.reserve(2); monitorables.push_back(vector<float>(1,0.)); monitorables.push_back(vector<float>(1,0.));
  return; }

  monitorables.resize(2,vector<float>());

  // Retrieve histogram contents and set monitorables
  vector<float>& peds = monitorables[0]; 
  vector<float>& noise = monitorables[1]; 
  for ( int ibin = 0; ibin < histos[0]->GetNbinsX(); ibin++ ) {
    peds.push_back( histos[0]->GetBinContent(ibin+1) );
    noise.push_back( histos[0]->GetBinError(ibin+1) );
  }

monitorables.reserve(2); monitorables.push_back(peds); monitorables.push_back(noise);
}



//   // Calculate calibration constants
//   peds.resize( temp.size() );
//   noise.resize( temp.size() );
//   for ( unsigned short ibin = 0; ibin < num.size(); ibin++ ) {
//     // Calculate pedestals
//     if ( num[ibin] ) { peds[ibin] = sum[ibin] / num[ibin]; }
//     else { peds[ibin] = 0; }
//     // Calculate noise
//     float square_of_mean = sum[ibin] * sum[ibin]; 
//     float mean_of_squares = 0.;
//     if ( num[ibin] ) { mean_of_squares = sum2[ibin] / num[ibin]; }
//     if ( mean_of_squares < square_of_mean ) { noise[ibin] = 0.; } // stat error here?...
//     else { noise[ibin] = sqrt( mean_of_squares - square_of_mean ); }
//   }
  

