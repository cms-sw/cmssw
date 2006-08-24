#include "DQM/SiStripCommissioningAnalysis/interface/OptoScanAnalysis.h"
#include "TProfile.h"
#include <iostream>
#include <cmath>

using namespace std;

// ----------------------------------------------------------------------------
// 
void OptoScanAnalysis::analysis( const TProfiles& histos, 
				 OptoScanAnalysis::Monitorables& mons ) { 
  
  //@@ use matt's method...
  if (1) { deprecated( histos, mons ); return; }

  // Target gain
  float target_gain = 0.8;
  float minimum_diff = 1.e9;
  
  // Iterate through four gain settings
  for ( uint16_t igain = 0; igain < 4; igain++ ) {
    
    // Select histos appropriate for gain setting
    TProfile* base = 0;
    TProfile* peak = 0;
    if      ( igain == 0 ) { base = histos.g0d0_; peak = histos.g0d1_; }
    else if ( igain == 1 ) { base = histos.g1d0_; peak = histos.g1d1_; }
    else if ( igain == 2 ) { base = histos.g2d0_; peak = histos.g2d1_; }
    else if ( igain == 3 ) { base = histos.g3d0_; peak = histos.g3d1_; }
    
    // Set gain setting in monitorables object
    mons.lldGain_ = igain;
    
    // Checks on whether histos exist
    if ( !base ) {
      cerr << "[" << __PRETTY_FUNCTION__ << "]"
	   << " NULL pointer to 'digital 0' histogram!"
	   << endl;
    }
    if ( !peak ) {
      cerr << "[" << __PRETTY_FUNCTION__ << "]"
	   << " NULL pointer to 'digital 1' histogram!"
	   << endl;
    }

  } // gain loop

}

// ----------------------------------------------------------------------------
// 
void OptoScanAnalysis::Monitorables::print( stringstream& ss ) { 
  ss << "OPTO SCAN Monitorables:" << "\n"
     << " LLD bias setting  : " << lldBias_ << "\n" 
     << " LLD gain setting  : " << lldGain_ << "\n"
     << " Measured gain     : " << gain_ << "\n" 
     << " Error on meas gain: " << error_ << "\n"
     << " Baseline     [adc]: " << base_ << "\n" 
     << " Tick peak    [adc]: " << peak_ << "\n" 
     << " Tick height  [adc]: " << height_ << "\n";
}

// ----------------------------------------------------------------------------
// 
void OptoScanAnalysis::TProfiles::print( stringstream& ss ) { 
  ss << "TProfile pointers:" << "\n"
     << " Gain 0, Digital 0: " << g0d0_ << "\n"
     << " Gain 0, Digital 1: " << g0d1_ << "\n"
     << " Gain 1, Digital 0: " << g1d0_ << "\n"
     << " Gain 1, Digital 1: " << g1d1_ << "\n"
     << " Gain 2, Digital 0: " << g2d0_ << "\n"
     << " Gain 2, Digital 1: " << g2d1_ << "\n"
     << " Gain 3, Digital 0: " << g3d0_ << "\n"
     << " Gain 3, Digital 1: " << g3d1_ << "\n";
}

// ----------------------------------------------------------------------------
// 
void OptoScanAnalysis::deprecated( const TProfiles& profs, 
				   OptoScanAnalysis::Monitorables& mons ) { 

  //   stringstream ss;
  //   const_cast<TProfiles&>(profs).print(ss);
  //   cout << "[" << __PRETTY_FUNCTION__ << "]" << endl
  //        << ss.str() << endl;
  
  vector<const TProfile*> histos; 
  vector<float> monitorables;
  
  float target_gain = 0.8;
  float diff_in_gain = 1.e9;
  for ( uint16_t igain = 0; igain < 4; igain++ ) {
    
    histos.clear();
    if ( igain == 0 ) {
      histos.push_back( const_cast<const TProfile*>(profs.g0d0_) );
      histos.push_back( const_cast<const TProfile*>(profs.g0d1_) );
    } else if ( igain == 1 ) {
      histos.push_back( const_cast<const TProfile*>(profs.g1d0_) );
      histos.push_back( const_cast<const TProfile*>(profs.g1d1_) );
    } else if ( igain == 2 ) {
      histos.push_back( const_cast<const TProfile*>(profs.g2d0_) );
      histos.push_back( const_cast<const TProfile*>(profs.g2d1_) );
    } else if ( igain == 3 ) {
      histos.push_back( const_cast<const TProfile*>(profs.g3d0_) );
      histos.push_back( const_cast<const TProfile*>(profs.g3d1_) );
    } 
    
    if ( !histos[0] ) {
      cerr << "[" << __PRETTY_FUNCTION__ << "]"
	   << " NULL pointer to base histo!" << endl;
      continue;
    }
    if ( !histos[1] ) {
      cerr << "[" << __PRETTY_FUNCTION__ << "]"
	   << " NULL pointer to tick histo!" << endl;
      continue;
    }

    monitorables.clear();
    analysis( histos, monitorables );

    if ( fabs(monitorables[0]-target_gain) < diff_in_gain ) {
      mons.lldGain_ = static_cast<uint16_t>( igain );
      mons.lldBias_ = static_cast<uint16_t>( monitorables[0] );
      mons.gain_    = monitorables[1];
      diff_in_gain = fabs(monitorables[0]-target_gain);
    }
    
  }
  
}

// ----------------------------------------------------------------------------
//
void OptoScanAnalysis::analysis( const vector<const TProfile*>& histos, 
				 vector<float>& monitorables ) {
  //edm::LogInfo("Commissioning|Analysis") << "[OptoScanAnalysis::analysis]";
  
  //extract root histograms
  //check 
  if (histos.size() != 2) { 
    //     edm::LogError("Commissioning|Analysis") 
    //       << "[OptoScanAnalysis::analysis]: Requires \"const vector<const TH1F*>& \" argument to have size 2. Actual size: " 
    //       << histos.size() << ". Monitorables set to 0."; 
    monitorables.push_back(0); monitorables.push_back(0);
  }

  //relabel
  const TProfile* base = histos[0];
  const TProfile* peak = histos[1];

  if ( !base ) {
    cerr << "[" << __PRETTY_FUNCTION__ << "]"
	 << " NULL pointer to base histo!" << endl;
  }
  if ( !peak ) {
    cerr << "[" << __PRETTY_FUNCTION__ << "]"
	 << " NULL pointer to peak histo!" << endl;
  }
  if ( !base || !peak ) { return; }

  //define utility objects
  vector<float> second_deriv_base; second_deriv_base.reserve(44); second_deriv_base.resize(44,0.);
  pair< int, int > slope_edges_base; slope_edges_base.first = 5; slope_edges_base.second = 5;
  
  //calculate the 2nd derivative of the histos and find slope edges

  for (int k=5;k<45;k++) {

    //checks

    // if (!base->GetBinContent(k)) {cout << "[OptoScanAnalysis::analysis]: Warning: Tick base has recorded value of 0 at bias: " << k - 1 << endl;}

    // if (!peak->GetBinContent(k)) { cout << "[OptoScanAnalysis::analysis]: Warning: Tick peak has recorded value of 0 at bias: " << k - 1 << endl;}

    second_deriv_base[k - 1] = 
      base->GetBinContent(k + 1) - 
      2*(base->GetBinContent(k)) + 
      base->GetBinContent(k - 1);

    //find bins containing 2 peaks in 2nd deriv i.e."slope edges"
    
    if (second_deriv_base[slope_edges_base.first] < second_deriv_base[k - 1]) { slope_edges_base.first = k - 1;}
    if (second_deriv_base[slope_edges_base.second] > second_deriv_base[k - 1]) { slope_edges_base.second = k - 1; }
  }

  //check

  if (slope_edges_base.first > slope_edges_base.second) {
    //        LogDebug("Commissioning|Analysis") 
    // 	 << "[OptoScanAnalysis::analysis]: Warning: Maximum second derivative of tick base occurs at higher bias: " 
    // 	 << slope_edges_base.first <<  " than the minimum: " << slope_edges_base.second << ".";
  }

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
    //        LogDebug("Commissioning|Analysis") 
    // 	 << "[OptoScanAnalysis::analysis]: Warning: No tick height found to match tick base at 70% off its maximum (> 10% difference between histograms)."; 
  }

  //Gain
  float gain = (slope_centerx_base - slope_centerx_peak) * slope_grad_base / 800.;

  //set monitorables
  monitorables.clear();
  monitorables.push_back(gain);
  monitorables.push_back(bias);

}
