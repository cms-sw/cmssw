#include "DQM/SiStripCommissioningAnalysis/interface/VPSPAnalysis.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TH1F.h"
#include <vector>
#include <cmath>

// -----------------------------------------------------------------------------
//
void VPSPAnalysis::analysis( const vector<const TH1F*>& histos, 
			      vector<unsigned short>& monitorables ) {
    edm::LogInfo("Commissioning|Analysis") << "[VPSPAnalysis::analysis]";

    //extract root histogram
    //check 
    if (histos.size() != 1) { edm::LogError("Commissioning|Analysis") << "[VPSPAnalysis::analysis]: Requires \"const vector<const TH1F*>& \" argument to have size 1. Actual size: " << histos.size() << ". Monitorables set to 0."; 
    
    monitorables.reserve(1); monitorables.push_back(0);
    return; }
    const TH1F* histo = histos[0];

 /*
  float top=0.;
  float bottom=1025.;
  for (int k=5;k<55;k++)
    {
      if (histo->GetBinContent(k) == 0) continue;
      if (histo->GetBinContent(k)>top)
	top=histo->GetBinContent(k);
      if (histo->GetBinContent(k)<bottom)
	bottom=histo->GetBinContent(k);
      
    }
  float optimum = bottom+1./3.*(top-bottom);

  for ( k2=5;k2<55;k2++) { if (histo->GetBinContent(k2)<optimum) break; }
  monitorables.optimumVPSP(k2 -1);//Added by M.W.
  */
  /////// or alternative method ...

// checks

    if ( (unsigned short)histo->GetNbinsX() < 58) {edm::LogError("Commissioning|Analysis") << "[VPSPAnalysis::analysis]: Warning: Insufficient number of bins in histogram. Number of bins: " << (unsigned short)histo->GetNbinsX() << ". Minimum required: 58.";

    monitorables.reserve(1); monitorables.push_back(0);
    return;}

  for ( unsigned int k = 1; k < 59; k++) {
    if ( (float)histo->GetBinContent(k) == 0. ) { LogDebug("Commissioning|Analysis") << "[VPSPAnalysis::analysis]: Warning: Baseline of 0 recorded at VPSP = " << k - 1 << ". Range required 0 - 58 inclusive.";}}

  vector<float> reduced_noise_histo; reduced_noise_histo.reserve(58); reduced_noise_histo.resize(58,0.);
  vector<float> second_deriv; second_deriv.reserve(54); second_deriv.resize(54,0.);
  pair< unsigned short, unsigned short > plateau_edges; plateau_edges.first = 0; plateau_edges.second = 0;

  //calculate a "reduced-noise" version of VPSP histogram @ Maybe only introduce this if noise > threshold value ???

  for (unsigned int k=4;k<56;k++) {// k represents bin number, starting at 1.
    for (unsigned int l = k -3; l < k + 4; l++) {
      reduced_noise_histo[k - 1] = (reduced_noise_histo[k - 1]*(l - k + 3) + (float)histo->GetBinContent(l)) / ( l - k + 4); //(int)histo->GetBinContent(k);
}}

  for (int k=5;k<55;k++) {
    
    //calculate the 2nd derivative of the reduced noise vector and relevent statistics
    
    second_deriv[k - 1] = reduced_noise_histo[k] - 2*(reduced_noise_histo[k-1]) + reduced_noise_histo[k-2];

    // Find "plateau edges"...using maximum/minimum
    
    if (second_deriv[plateau_edges.first] > second_deriv[k - 1]) {plateau_edges.first = k - 1;}
    if (second_deriv[plateau_edges.second] < second_deriv[k - 1]) {plateau_edges.second = k - 1;}

  }

  // median...

  vector<float> sorted_second_deriv; sorted_second_deriv.reserve(second_deriv.size());
  sorted_second_deriv = second_deriv;
  sort(sorted_second_deriv.begin(), sorted_second_deriv.end());
  float median_2D_90pc = sorted_second_deriv[(unsigned short)(sorted_second_deriv.size()*.9)];
  float median_2D_10pc = sorted_second_deriv[(unsigned short)(sorted_second_deriv.size()*.1)];

  //check minimum 2nd derivative VPSP < maximum 2nd derivative VPSP

  if (plateau_edges.first > plateau_edges.second) {LogDebug("Commissioning|Analysis") << "[VPSPAnalysis::analysis]: Warning: Minimum second derivative found at higher VPSP value than the maximum. Min VPSP = " << plateau_edges.first << " and Max VPSP = " << plateau_edges.second << ".";}


 // loop bins and find mean and sigma of noise of second deriv avoiding the peaks

 float mean_2D_noise = 0.;
 float mean2_2D_noise = 0.;
 unsigned short count = 0;

for (int k=5;k<55;k++) {
  if ((second_deriv[k - 1] < (median_2D_90pc)) && (second_deriv[k - 1] > (median_2D_10pc))) { mean_2D_noise +=second_deriv[k - 1]; mean2_2D_noise += (second_deriv[k - 1] * second_deriv[k - 1]); count++;}
}

 if (count) {mean_2D_noise = mean_2D_noise/ (float)count; mean2_2D_noise = mean2_2D_noise / (float)count;}

float sigma_2D_noise = sqrt(fabs(mean_2D_noise * mean_2D_noise - mean2_2D_noise));

//check peaks ARE above mean of the noise +- 2*sigma 

 if (second_deriv[plateau_edges.first] > (mean_2D_noise - 2*sigma_2D_noise)) { LogDebug("Commissioning|Analysis") << "[VPSPAnalysis::analysis]: Warning: noise of second derivative large. Minimum second derivative = " << second_deriv[plateau_edges.first] << ". Mean and S.D. of 2nd derivative noise are " << mean_2D_noise << " and " << sigma_2D_noise << " respectively."; }

 if (second_deriv[plateau_edges.second] < (mean_2D_noise + 2*sigma_2D_noise)) { LogDebug("Commissioning|Analysis") << "[VPSPAnalysis::analysis]: Warning: noise of second derivative large. Maximum second derivative = " << second_deriv[plateau_edges.second] << ". Mean and S.D. of 2nd derivative noise are " << mean_2D_noise << " and " << sigma_2D_noise << " respectively."; }

//find positions where 2nd deriv peaks flatten

 while ((second_deriv[plateau_edges.first] < (mean_2D_noise - 2*sigma_2D_noise)) && (plateau_edges.first > 5)) { plateau_edges.first--;
}
 while ((second_deriv[plateau_edges.second] > (mean_2D_noise + 2*sigma_2D_noise)) && (plateau_edges.first < 55)) { plateau_edges.second++;}

// locate optimum VPSP value

  float top_mean = 0, bottom_mean = 0;
  for ( unsigned short m = 4; m < (plateau_edges.first +1); m++ ) {
    top_mean = (top_mean*(m - 4) + (int)histo->GetBinContent(m + 1))/ (m - 3);}
  
  for ( unsigned short m = plateau_edges.second; m < 55; m++ ) { 
    bottom_mean = ((bottom_mean* (m - plateau_edges.second) ) + (int)histo->GetBinContent(m + 1))/ (m - plateau_edges.second + 1);}

  float optimum = bottom_mean + (top_mean - bottom_mean) * 1./3.;
  float gradient = (float)((int)histo->GetBinContent(plateau_edges.second + 1) - (int)histo->GetBinContent(plateau_edges.first + 1)) / (float)(plateau_edges.second - plateau_edges.first);
  
  unsigned short vpsp = (unsigned short)((optimum - (unsigned short)histo->GetBinContent(plateau_edges.first + 1)) / gradient) + plateau_edges.first;
  
//set monitorables
  monitorables.clear();
  monitorables.reserve(1);
  monitorables.push_back(vpsp);

}


