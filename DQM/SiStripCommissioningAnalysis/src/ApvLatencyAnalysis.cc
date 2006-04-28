#include "DQM/SiStripCommissioningAnalysis/interface/ApvLatencyAnalysis.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TProfile.h"
#include <vector>
#include <cmath>

// -----------------------------------------------------------------------------
//
void ApvLatencyAnalysis::analysis( const vector<const TProfile*>& histos, 
			      vector<unsigned short>& monitorables ) {
    edm::LogInfo("Commissioning|Analysis") << "[ApvLatencyAnalysis::analysis]";

    //extract root histogram
    //check 
  if (histos.size() != 1) { edm::LogError("Commissioning|Analysis") << "[ApvLatencyAnalysis::analysis]: Requires \"const vector<const TH1F*>& \" argument to have size 1. Actual size: " << histos.size() << ". Monitorables set to 0."; 
  
  monitorables.push_back(0);
  return; }
  const TProfile* histo = histos[0];

    //monitorable
    unsigned short latency;

 vector<unsigned short> binContent; binContent.reserve((unsigned short)histo->GetNbinsX()); binContent.resize((unsigned short)histo->GetNbinsX(), 0);

 for (unsigned short k = 0; k < (unsigned short)histo->GetNbinsX(); k++) { // k is bin number

//fill vector with histogram contents
    binContent.push_back((unsigned int)(histo->GetBinContent(k)));}

 //calculate median
  
 sort(binContent.begin(), binContent.end());
 
 //calculate mean and mean2 of the readout within cutoffs
 
 float meanNoise = 0.;//M.W method
 float mean2Noise = 0.;
 
 for (unsigned short k = (unsigned short)(binContent.size()*.1); k < (unsigned short)(binContent.size()*.9); k++) {
   meanNoise += binContent[k];
   mean2Noise += binContent[k]*binContent[k];;
 }
 
 meanNoise = meanNoise * binContent.size() * 0.8;
 mean2Noise = mean2Noise * binContent.size() * 0.8;
 float sigmaNoise = sqrt(fabs(meanNoise*meanNoise - mean2Noise));
 
 //loop to look for signal > 5* sigma_noise
 unsigned short count = 0;
 unsigned short maxlatency = 0;
 unsigned int maxhits = 0;
 
 for (unsigned short k = 1; k < ((unsigned short)histo->GetNbinsX() + 1); k++) { // k is bin number
   if (histo->GetBinContent((Int_t)k) > maxhits) maxlatency = k - 1;
   if ((float)histo->GetBinContent((Int_t)k) > (meanNoise + 5 * sigmaNoise)) { 
     latency = k - 1; count++;
   }
 }
 
 if (!count) {LogDebug("Commissioning|Analysis") << "[ApvLatencyAnalysis::analysis]: Warning: no signal found > mean + 5*sigma(noise). Returning latency of highest number of recorded hits." << endl;
 latency = maxlatency;}
 
 if (count > 1) {LogDebug("Commissioning|Analysis") << "[ApvLatencyAnalysis::analysis]: Warning: more than one signal found > mean + 5*sigma(noise). Returning latency of highest number of recorded hits.";
 latency = maxlatency;}

 //set monitorables
 monitorables.clear();
 monitorables.push_back(latency);
}
