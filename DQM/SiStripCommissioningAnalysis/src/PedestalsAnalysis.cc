#include "DQM/SiStripCommissioningAnalysis/interface/PedestalsAnalysis.h"
#include "TH1F.h"
#include <iostream>
#include <vector>
#include <cmath>

// -----------------------------------------------------------------------------
//
void PedestalsAnalysis::analysis( const PedestalsHistograms& histos, 
				  PedestalsMonitorables& monitorables ) {
  cout << "[PedestalsAnalysis::analysis]" << endl;
  
  // Retrieve histogram contents
  vector<float> peds; 
  vector<float> noise; 
  for ( int ibin = 0; ibin < histos.peds()->GetNbinsX(); ibin++ ) {
    peds.push_back( histos.peds()->GetBinContent(ibin+1) );
    noise.push_back( histos.peds()->GetBinError(ibin+1) );
  }
  
  // Write calibration constansts to Monitorables object
  monitorables.rawPeds( peds );
  monitorables.rawNoise( noise );
  
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
  

