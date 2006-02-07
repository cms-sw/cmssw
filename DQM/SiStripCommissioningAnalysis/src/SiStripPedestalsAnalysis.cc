#include "DQM/SiStripCommissioningAnalysis/interface/SiStripPedestalsAnalysis.h"
#include "TH1F.h"
#include <iostream>

// -----------------------------------------------------------------------------
//
void SiStripPedestalsAnalysis::histoAnalysis( const PedestalHistograms& histograms, 
					      PedestalMonitorables& monitorables ) {
  std::cout << "[SiStripPedestalsAnalysis::histoAnalysis]" << std::endl;
  
  //@@ ANALYSIS HERE! EG:
  TH1F* histo = histograms.raw();
  std::vector<float> peds; peds.clear();
  std::vector<float> noise; noise.clear();
  for ( int ibin = 0; ibin < histo->GetNbinsX(); ibin++ ) {
    peds.push_back( histo->GetBinContent(ibin+1) );
    noise.push_back( histo->GetBinError(ibin+1) );
  }
  monitorables.rawPedestals( peds );
  monitorables.rawNoise( noise );
  // THAT'S IT!
  
}


