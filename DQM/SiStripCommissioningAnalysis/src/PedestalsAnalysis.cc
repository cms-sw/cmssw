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
  vector<float> sum2; 
  vector<float> sum;
  vector<float> num;
  for ( int ibin = 0; ibin < histos.peds().numOfEntries()->GetNbinsX(); ibin++ ) {
    sum2.push_back( histos.peds().sumOfSquares()->GetBinContent(ibin+1) );
    sum.push_back( histos.peds().sumOfContents()->GetBinContent(ibin+1) );
    num.push_back( histos.peds().numOfEntries()->GetBinContent(ibin+1) );
  }
  
  // Calculate calibration constants
  vector<float> peds; 
  vector<float> noise; 
  peds.resize( num.size() );
  noise.resize( num.size() );
  for ( unsigned short ibin = 0; ibin < num.size(); ibin++ ) {
    // Calculate pedestals
    if ( num[ibin] ) { peds[ibin] = sum[ibin] / num[ibin]; }
    else { peds[ibin] = 0; }
    // Calculate noise
    float square_of_mean = sum[ibin] * sum[ibin]; 
    float mean_of_squares = 0.;
    if ( num[ibin] ) { mean_of_squares = sum2[ibin] / num[ibin]; }
    if ( mean_of_squares < square_of_mean ) { noise[ibin] = 0.; } // stat error here?...
    else { noise[ibin] = sqrt( mean_of_squares - square_of_mean ); }
  }
  
  // Write calibration constansts to Monitorables object
  monitorables.rawPeds( peds );
  monitorables.rawNoise( noise );
  
}


