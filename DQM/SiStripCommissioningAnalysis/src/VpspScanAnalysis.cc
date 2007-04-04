#include "DQM/SiStripCommissioningAnalysis/interface/VpspScanAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TProfile.h"
#include "TH1.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
VpspScanAnalysis::VpspScanAnalysis( const uint32_t& key )
  : CommissioningAnalysis(key,"VPSP SCAN"),
    vpsp0_(sistrip::invalid_), 
    vpsp1_(sistrip::invalid_),
    hVpsp0_(0,""), 
    hVpsp1_(0,"")
{;}

// -----------------------------------------------------------------------------
//
VpspScanAnalysis::VpspScanAnalysis()
  : CommissioningAnalysis("VPSP SCAN"),
    vpsp0_(sistrip::invalid_), 
    vpsp1_(sistrip::invalid_),
    hVpsp0_(0,""), 
    hVpsp1_(0,"")
{;}

// ----------------------------------------------------------------------------
// 
void VpspScanAnalysis::print( std::stringstream& ss, uint32_t not_used ) { 
  header( ss );
  ss << " VPSP setting APV0: " << vpsp0_ << "\n" 
     << " VPSP setting APV1: " << vpsp1_ << "\n" ;
}

// -----------------------------------------------------------------------------
//
void VpspScanAnalysis::reset() {
  vpsp0_ = sistrip::invalid_;
  vpsp1_ = sistrip::invalid_;
  hVpsp0_ = Histo(0,"");
  hVpsp1_ = Histo(0,"");
}

// ----------------------------------------------------------------------------
// 
void VpspScanAnalysis::extract( const std::vector<TH1*>& histos ) { 

  // Check
  if ( histos.size() != 2 ) {
    edm::LogWarning(mlCommissioning_)
      << "[" << myName() << "::" << __func__ << "]"
      << " Unexpected number of histograms: " 
      << histos.size();
  }
  
  // Extract FED key from histo title
  if ( !histos.empty() ) extractFedKey( histos.front() );

  // Extract
  std::vector<TH1*>::const_iterator ihis = histos.begin();
  for ( ; ihis != histos.end(); ihis++ ) {
    
    // Check pointer
    if ( !(*ihis) ) {
      edm::LogWarning(mlCommissioning_) 
	<< "[" << myName() << "::" << __func__ << "]"
	<< " NULL pointer to histogram!";
      continue;
    }
    
    // Check name
    SiStripHistoTitle title( (*ihis)->GetName() );
    if ( title.runType() != sistrip::VPSP_SCAN ) {
      edm::LogWarning(mlCommissioning_)
	<< "[" << myName() << "::" << __func__ << "]"
	<< " Unexpected commissioning task: "
	<< SiStripEnumsAndStrings::runType(title.runType());
      continue;
    }
    
    // Extract APV number
    uint16_t apv = sistrip::invalid_; 
    if ( title.extraInfo().find(sistrip::apv_) != std::string::npos ) {
      std::stringstream ss;
      ss << title.extraInfo().substr( title.extraInfo().find(sistrip::apv_) + sistrip::apv_.size(), 1 );
      ss >> std::dec >> apv;
    }
    
    // Store vpsp scan histos
    if ( apv == 0 ) { 
      hVpsp0_.first = *ihis; 
      hVpsp0_.second = (*ihis)->GetName();
    } else if ( apv == 1 ) { 
      hVpsp1_.first = *ihis; 
      hVpsp1_.second = (*ihis)->GetName();
    } else {
      edm::LogWarning(mlCommissioning_) 
	<< "[" << myName() << "::" << __func__ << "]"
	<< " Unexpected APV number! (" << apv << ")";
    }
    
  }

}

// -----------------------------------------------------------------------------
//
void VpspScanAnalysis::analyse() {
  deprecated(); //@@ use matt's method...
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
//
void VpspScanAnalysis::deprecated() {
  
  std::vector<const TProfile*> histos; 
  std::vector<unsigned short> monitorables;
  for ( uint16_t iapv = 0; iapv < 2; iapv++ ) {
    
    histos.clear();
    if ( iapv == 0 ) {
      histos.push_back( const_cast<const TProfile*>( dynamic_cast<TProfile*>(hVpsp0_.first) ) );
    } else if ( iapv == 1 ) {
      histos.push_back( const_cast<const TProfile*>( dynamic_cast<TProfile*>(hVpsp1_.first) ) );
    } 
    if ( !histos[0] ) {
      edm::LogWarning(mlCommissioning_)
	<< "[" << myName() << "::" << __func__ << "]"
	<< " NULL pointer to VPSP histo for APV" << iapv;
      continue;
    }
    
    monitorables.clear();
    anal( histos, monitorables );
    
    if ( iapv == 0 ) {
      vpsp0_ = monitorables[0];
    } else if ( iapv == 1 ) {
      vpsp1_ = monitorables[0];
    }
    
  }

}

// -----------------------------------------------------------------------------
//
void VpspScanAnalysis::anal( const std::vector<const TProfile*>& histos, 
			     std::vector<unsigned short>& monitorables ) {
  //LogDebug("Commissioning|Analysis") << "[VpspScanAnalysis::analysis]";

  //extract root histogram
  //check 
  if (histos.size() != 1) { 
    //     edm::LogWarning("Commissioning|Analysis") << "[VpspScanAnalysis::analysis]: Requires \"const std::vector<const TH1F*>& \" argument to have size 1. Actual size: " << histos.size() << ". Monitorables set to 0."; 
    monitorables.push_back(0);
    return; 
  }
  const TProfile* histo = histos[0];

// checks

    if ( (unsigned short)histo->GetNbinsX() < 58) {
//       edm::LogWarning("Commissioning|Analysis") << "[VpspScanAnalysis::analysis]: Warning: Insufficient number of bins in histogram. Number of bins: " << (unsigned short)histo->GetNbinsX() << ". Minimum required: 58.";

    monitorables.push_back(0);
    return;}

  for ( unsigned int k = 1; k < 59; k++) {
    if ( (float)histo->GetBinContent(k) == 0. ) { 
//       LogDebug("Commissioning|Analysis") << "[VpspScanAnalysis::analysis]: Warning: Baseline of 0 recorded at VPSP = " << k - 1 << ". Range required 0 - 58 inclusive.";
    }
  }

  std::vector<float> reduced_noise_histo; reduced_noise_histo.reserve(58); reduced_noise_histo.resize(58,0.);
  std::vector<float> second_deriv; second_deriv.reserve(54); second_deriv.resize(54,0.);
  std::pair< unsigned short, unsigned short > plateau_edges; plateau_edges.first = 0; plateau_edges.second = 0;

  //calculate a "reduced-noise" version of VPSP histogram @@ Maybe only introduce this if noise > threshold value ???

  for (unsigned int k=4;k<56;k++) {// k represents bin number, starting at 1.
    for (unsigned int l = k -3; l < k + 4; l++) {
      reduced_noise_histo[k - 1] = (reduced_noise_histo[k - 1]*(l - k + 3) + (float)histo->GetBinContent(l)) / ( l - k + 4); //(int)histo->GetBinContent(k);
}}

  for (int k=5;k<55;k++) {
    
    //calculate the 2nd derivative of the reduced noise std::vector and relevent statistics
    
    second_deriv[k - 1] = reduced_noise_histo[k] - 2*(reduced_noise_histo[k-1]) + reduced_noise_histo[k-2];

    // Find "plateau edges"...using maximum/minimum
    
    if (second_deriv[plateau_edges.first] > second_deriv[k - 1]) {plateau_edges.first = k;}
    if (second_deriv[plateau_edges.second] < second_deriv[k - 1]) {plateau_edges.second = k;}

  }

  // median...

  std::vector<float> sorted_second_deriv; sorted_second_deriv.reserve(second_deriv.size());
  sorted_second_deriv = second_deriv;
  sort(sorted_second_deriv.begin(), sorted_second_deriv.end());
  float median_2D_90pc = sorted_second_deriv[(unsigned short)(sorted_second_deriv.size()*.9)];
  float median_2D_10pc = sorted_second_deriv[(unsigned short)(sorted_second_deriv.size()*.1)];

  //check minimum 2nd derivative VPSP < maximum 2nd derivative VPSP

  if (plateau_edges.first > plateau_edges.second) {
//     LogDebug("Commissioning|Analysis") << "[VpspScanAnalysis::analysis]: Warning: Minimum second derivative found at higher VPSP value than the maximum. Min VPSP = " << plateau_edges.first << " and Max VPSP = " << plateau_edges.second << ".";
  }


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

 if (second_deriv[plateau_edges.first] > (mean_2D_noise - 2*sigma_2D_noise)) { 
 //   LogDebug("Commissioning|Analysis") << "[VpspScanAnalysis::analysis]: Warning: noise of second derivative large. Minimum second derivative = " << second_deriv[plateau_edges.first] << ". Mean and S.D. of 2nd derivative noise are " << mean_2D_noise << " and " << sigma_2D_noise << " respectively."; 
 }

 if (second_deriv[plateau_edges.second] < (mean_2D_noise + 2*sigma_2D_noise)) { 
 //   LogDebug("Commissioning|Analysis") << "[VpspScanAnalysis::analysis]: Warning: noise of second derivative large. Maximum second derivative = " << second_deriv[plateau_edges.second] << ". Mean and S.D. of 2nd derivative noise are " << mean_2D_noise << " and " << sigma_2D_noise << " respectively."; 
 }

//find positions where 2nd deriv peaks flatten

 while ((second_deriv[plateau_edges.first] < (mean_2D_noise - 2*sigma_2D_noise)) && (plateau_edges.first > 5)) { plateau_edges.first--;
}
 while ((second_deriv[plateau_edges.second] > (mean_2D_noise + 2*sigma_2D_noise)) && (plateau_edges.first < 55)) { plateau_edges.second++;}

// locate optimum VPSP value
 float top_mean = 0., bottom_mean = 0.;
 for ( unsigned short m = 5; m < plateau_edges.first; m++ ) {
   top_mean = (top_mean*(m - 5) + histo->GetBinContent(m))/ (m - 4);}
 
 for ( unsigned short m = plateau_edges.second; m < 56; m++ ) { 
   bottom_mean = ((bottom_mean* (m - plateau_edges.second) ) + histo->GetBinContent(m))/ (m - plateau_edges.second+1);}
 float optimum = bottom_mean + (top_mean - bottom_mean) * 1./3.;
 
  /////// or alternative method ..
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
  float opt = bottom+1./3.*(top-bottom);
  LogTrace(mlCommissioning_) << opt << ";" << optimum;
  */

  unsigned short vpsp;
  for (vpsp=plateau_edges.first; vpsp<plateau_edges.second; vpsp++) { if (histo->GetBinContent(vpsp)<optimum) break; }
  
//set monitorables
  monitorables.clear();
  monitorables.push_back(vpsp);

}


