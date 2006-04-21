#include "DQM/SiStripCommissioningAnalysis/interface/BiasGainAnalysis.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TH1F.h"
#include <vector>
#include <cmath>

// ----------------------------------------------------------------------------

void BiasGainAnalysis::analysis( const vector<const TH1F*>& histos, 
			      vector<float>& monitorables ) {
  edm::LogInfo("Commissioning|Analysis") << "[BiasGainAnalysis::analysis]";
  
  //extract root histograms
  //check 
  if (histos.size() != 2) { edm::LogError("Commissioning|Analysis") << "[BiasGainAnalysis::analysis]: Requires \"const vector<const TH1F*>& \" argument to have size 2. Actual size: " << histos.size() << ". Monitorables set to 0."; 
  
  monitorables.reserve(2); monitorables.push_back(0); monitorables.push_back(0);
  return; }

  //relabel
  const TH1F* base = histos[0];
  const TH1F* peak = histos[1];

  //define utility objects
  vector<float> second_deriv_base; second_deriv_base.reserve(44); second_deriv_base.resize(44,0.);
  pair< int, int > slope_edges_base; slope_edges_base.first = 5; slope_edges_base.second = 5;
  
  //calculate the 2nd derivative of the histos and find slope edges

     for (int k=5;k<45;k++) {

       //checks

       // if (!base->GetBinContent(k)) {cout << "[BiasGainAnalysis::analysis]: Warning: Tick base has recorded value of 0 at bias: " << k - 1 << endl;}

       // if (!peak->GetBinContent(k)) { cout << "[BiasGainAnalysis::analysis]: Warning: Tick peak has recorded value of 0 at bias: " << k - 1 << endl;}

    second_deriv_base[k - 1] = base->GetBinContent(k + 1) - 2*(base->GetBinContent(k)) + base->GetBinContent(k - 1);

    //find bins containing 2 peaks in 2nd deriv i.e."slope edges"
    
    if (second_deriv_base[slope_edges_base.first] < second_deriv_base[k - 1]) { slope_edges_base.first = k - 1;}
    if (second_deriv_base[slope_edges_base.second] > second_deriv_base[k - 1]) { slope_edges_base.second = k - 1; }
     }

     //check

     if (slope_edges_base.first > slope_edges_base.second) {LogDebug("Commissioning|Analysis") << "[BiasGainAnalysis::analysis]: Warning: Maximum second derivative of tick base occurs at higher bias: " << slope_edges_base.first <<  " than the minimum: " << slope_edges_base.second << ".";}

     //CALCULATE BIAS
     //find position of - first point after 2nd deriv max below 0.2 x max (for base) - and - first point before 2nd deriv min above 0.2 x min (for base and peak).

     while (fabs(second_deriv_base[slope_edges_base.second]) > fabs(0.2 * slope_edges_base.second)) {slope_edges_base.second--;}
     while (fabs(second_deriv_base[slope_edges_base.first]) > fabs(0.2 * slope_edges_base.first)) {slope_edges_base.first++;}
  
     float bias = (float)slope_edges_base.first;

     //CALCULATE GAIN
     //Find bias where the peak/base is closest to 300 (tunable)...

     float slope_grad_base = (float) (base->GetBinContent(slope_edges_base.second) - base->GetBinContent(slope_edges_base.first)) / (float)(slope_edges_base.second - slope_edges_base.first);

     unsigned short slope_centerx_base = 0, slope_centerx_peak = 0;

     for (unsigned short baseBias = 0; baseBias < 45; baseBias++) {
       if (fabs(base->GetBinContent((Int_t)(baseBias + 1)) - 300) < fabs(base->GetBinContent((Int_t)(slope_centerx_base)) - 300)) slope_centerx_base = baseBias;}

     for (unsigned short peakBias = 0; peakBias < 45; peakBias++) {
       if (fabs(peak->GetBinContent((Int_t)(peakBias + 1)) - 300) < fabs(peak->GetBinContent((Int_t)(slope_centerx_peak)) - 300)) slope_centerx_peak = peakBias;}
 
     //check
     if (((peak->GetBinContent((Int_t)(slope_centerx_peak)) - base->GetBinContent((Int_t)(slope_centerx_base)))/ (float)base->GetBinContent((Int_t)(slope_centerx_base))) > 0.1) { 

       LogDebug("Commissioning|Analysis") << "[BiasGainAnalysis::analysis]: Warning: No tick height found to match tick base at 70% off its maximum (> 10% difference between histograms)."; }

     //Gain
     float gain = (slope_centerx_base - slope_centerx_peak) * slope_grad_base / 800.;

     //set monitorables
     monitorables.clear();
     monitorables.reserve(2);
     monitorables.push_back(gain);
     monitorables.push_back(bias);
}

