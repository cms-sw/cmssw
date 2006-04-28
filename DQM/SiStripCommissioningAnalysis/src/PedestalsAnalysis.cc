#include "DQM/SiStripCommissioningAnalysis/interface/PedestalsAnalysis.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TProfile.h"
#include <vector>
#include <cmath>

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


