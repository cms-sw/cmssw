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
  : CommissioningAnalysis(key,"VpspScanAnalysis"),
    vpsp_(2,sistrip::invalid_), 
    adcLevel_(2,sistrip::invalid_),
    fraction_(2,sistrip::invalid_),
    topEdge_(2,sistrip::invalid_),
    bottomEdge_(2,sistrip::invalid_),
    topLevel_(2,sistrip::invalid_),
    bottomLevel_(2,sistrip::invalid_),
    hVpsp0_(0,""), 
    hVpsp1_(0,"")
{;}

// -----------------------------------------------------------------------------
//
VpspScanAnalysis::VpspScanAnalysis()
  : CommissioningAnalysis("VpspScanAnalysis"),
    vpsp_(2,sistrip::invalid_), 
    adcLevel_(2,sistrip::invalid_),
    fraction_(2,sistrip::invalid_),
    topEdge_(2,sistrip::invalid_),
    bottomEdge_(2,sistrip::invalid_),
    topLevel_(2,sistrip::invalid_),
    bottomLevel_(2,sistrip::invalid_),
    hVpsp0_(0,""), 
    hVpsp1_(0,"")
{;}

// ----------------------------------------------------------------------------
// 
void VpspScanAnalysis::print( std::stringstream& ss, uint32_t iapv ) { 
  if ( iapv == 1 || iapv == 2 ) { iapv--; }
  else { iapv = 0; }
  header( ss );
  ss << " Monitorables for APV number     : " << iapv;
  if ( iapv == 0 ) { ss << " (first of pair)"; }
  else if ( iapv == 1 ) { ss << " (second of pair)"; } 
  ss << std::endl;
  ss << " VPSP setting       : " << vpsp_[iapv] << std::endl 
     << " Signal level [ADC] : " << adcLevel_[iapv] << std::endl
     << " Fraction [%]       : " << fraction_[iapv] << std::endl
     << " Top edge [bin]     : " << topEdge_[iapv] << std::endl
     << " Bottom edge [bin]  : " << bottomEdge_[iapv] << std::endl
     << " Top level [ADC]    : " << topLevel_[iapv] << std::endl
     << " Bottom level [ADC] : " << bottomLevel_[iapv] << std::endl;
}

// -----------------------------------------------------------------------------
//
void VpspScanAnalysis::reset() {
  vpsp_ = VInt(2,sistrip::invalid_);
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

  //@@ use matt's method...
  deprecated(); 
  
}

// -----------------------------------------------------------------------------
//
void VpspScanAnalysis::deprecated() {
  
  std::vector<const TProfile*> histos; 
  std::vector<uint16_t> monitorables;
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

    if ( 0 ) { anal( histos, monitorables ); }
    else {

      int first = 0;
      float top = 0.;
      for ( int k = 5; k < 55; k++ ) {
	if ( histos[0]->GetBinContent(k) == 0 ) { continue; }
	if ( histos[0]->GetBinContent(k) >= top ) { 
	  first = k; 
	  top = histos[0]->GetBinContent(k); 
	}
      }

      int last = 60;
      float bottom = 1025.;
      for ( int k = 55; k > 5; k-- ) {
	if ( histos[0]->GetBinContent(k) == 0 ) { continue; }
	if ( histos[0]->GetBinContent(k) <= bottom ) { 
	  last = k; 
	  bottom = histos[0]->GetBinContent(k); 
	}
      }
      
      float opt = bottom + ( top - bottom ) * 1./3.;
      uint16_t vpsp;
      for ( vpsp = 5; vpsp < 55; vpsp++ ) { 
	if ( histos[0]->GetBinContent(vpsp) < opt ) { break; }
      }
      
      monitorables.push_back(vpsp);
      monitorables.push_back(static_cast<uint16_t>(opt));
      monitorables.push_back(65535);
      monitorables.push_back(first);
      monitorables.push_back(last);
      monitorables.push_back(static_cast<uint16_t>(top));
      monitorables.push_back(static_cast<uint16_t>(bottom));

    }
    
    vpsp_[iapv]        = monitorables[0];
    adcLevel_[iapv]    = monitorables[1];
    fraction_[iapv]    = monitorables[2];
    topEdge_[iapv]     = monitorables[3];
    bottomEdge_[iapv]  = monitorables[4];
    topLevel_[iapv]    = monitorables[5];
    bottomLevel_[iapv] = monitorables[6];
   
  }

}

// -----------------------------------------------------------------------------
//
void VpspScanAnalysis::anal( const std::vector<const TProfile*>& histos, 
			     std::vector<uint16_t>& monitorables ) {

  // Level of baseline
  const float fractional_baseline_level = 1./3.;

  // Check for number of histos
  if ( histos.size() != 1 ) { 
    edm::LogWarning(mlCommissioning_)
      << "[VpspScanAnalysis::" << __func__ << "]"
      << " Unexpected number of histos! (" << histos.size()
      << ") Should be 1."; 
    monitorables.push_back(0);
    return; 
  }

  // Check number of bins
  const TProfile* histo = histos[0];
  if ( static_cast<uint16_t>(histo->GetNbinsX()) < 58 ) {
    edm::LogWarning(mlCommissioning_)
      << "[VpspScanAnalysis::" << __func__ << "]"
      << " Insufficient number of bins in histogram ("
      << static_cast<uint16_t>(histo->GetNbinsX())
      << ") Minimum required is 58";
    monitorables.push_back(0);
    return;
  }

  // Check for null contents 
  for ( uint16_t k = 1; k < 59; k++ ) {
    if ( !histo->GetBinContent(k) ) { 
      LogDebug(mlCommissioning_) 
	<< "[VpspScanAnalysis::" << __func__ << "]"
	<< " Null baseline level found for VPSP setting " << (k-1);
    }
  }

  // Calculate a "reduced-noise" version of VPSP histogram 
  std::vector<float> reduced_noise_histo; 
  std::vector<float> second_deriv; 
  reduced_noise_histo.resize(58,0.);
  second_deriv.resize(54,0.);
  std::stringstream ss1;
  ss1 << " reduced_noise_histo ";
  for ( uint32_t k = 4; k < 56; k++ ) { // k is bin number
    for ( uint32_t l = k-3; l < k+4; l++ ) {
      reduced_noise_histo[k-1] = 
	( reduced_noise_histo[k-1]*(l-k+3) + 
	  static_cast<float>(histo->GetBinContent(l)) ) / 
	(l-k+4); 
      ss1 << k-1 << "/" << reduced_noise_histo[k-1] << " ";
    }
  }
  LogDebug(mlCommissioning_) << ss1.str();
  
  // Calculate 2nd derivative and find "plateau edges"
  std::stringstream ss2;
  ss2 << " second_deriv ";
  std::pair<uint16_t,uint16_t> plateau_edges; 
  plateau_edges.first = 0; 
  plateau_edges.second = 0;
  for ( uint16_t k = 5; k < 55; k++ ) {
    second_deriv[k-1] = reduced_noise_histo[k] - 2*(reduced_noise_histo[k-1]) + reduced_noise_histo[k-2];
    ss2 << k-1 << "/" << second_deriv[k-1] << " ";
    if ( second_deriv[plateau_edges.first] > second_deriv[k-1] ) { plateau_edges.first = k; ss2 << " first! "; }
    if ( second_deriv[plateau_edges.second] < second_deriv[k-1] ) { plateau_edges.second = k; ss2 << " second! "; }
  }
  ss2 << std::endl << plateau_edges.first << " " << plateau_edges.second;
  LogDebug(mlCommissioning_) << ss2.str();

  // Calc median
  std::vector<float> sorted_second_deriv; 
  sorted_second_deriv = second_deriv;
  sort( sorted_second_deriv.begin(), sorted_second_deriv.end() );
  float median_2D_90pc = sorted_second_deriv[ static_cast<uint16_t>(0.9*sorted_second_deriv.size()) ];
  float median_2D_10pc = sorted_second_deriv[ static_cast<uint16_t>(0.1*sorted_second_deriv.size()) ];
  std::stringstream ss3;
  ss3 << " median " << sorted_second_deriv.size() << " " 
      << median_2D_10pc << " " << median_2D_90pc;
  LogDebug(mlCommissioning_) << ss3.str();
  
  // Check minimum 2nd derivative VPSP < maximum 2nd derivative VPSP
  if ( plateau_edges.first > plateau_edges.second ) {
    LogDebug(mlCommissioning_) 
      << "[VpspScanAnalysis::" << __func__ << "]"
      << " Minimum second derivative (" << plateau_edges.first
      << ") found at higher VPSP value than the maximum (" << plateau_edges.second
      << ")!";
  }
  
  // Find mean and sigma of noise of second deriv, avoiding the peaks
  float mean_2D_noise = 0.;
  float mean2_2D_noise = 0.;
  uint16_t count = 0;
  for ( uint16_t k = 5; k < 55; k++ ) {
    if ( second_deriv[k-1] < median_2D_90pc && 
	 second_deriv[k-1] > median_2D_10pc ) { 
      mean_2D_noise += second_deriv[k-1]; 
      mean2_2D_noise += second_deriv[k-1] * second_deriv[k-1]; 
      count++;
    }
  }
  
  if ( count ) {
    mean_2D_noise = mean_2D_noise / static_cast<float>(count); 
    mean2_2D_noise = mean2_2D_noise / static_cast<float>(count);
  }
  float sigma_2D_noise = sqrt( fabs( mean_2D_noise * mean_2D_noise - mean2_2D_noise ) );
  std::stringstream ss4;
  ss4 << " noise " << mean_2D_noise << " " 
      << mean2_2D_noise << " " << count << " " << sigma_2D_noise;
  LogDebug(mlCommissioning_) << ss4.str();
  
  // Check first peak is above mean of the noise +/- 2*sigma 
  if ( second_deriv[plateau_edges.first] > mean_2D_noise - 2*sigma_2D_noise ) { 
    LogDebug(mlCommissioning_) 
      << "[VpspScanAnalysis::" << __func__ << "]" 
      << " Noise of second derivative too large (" 
      << second_deriv[plateau_edges.first] 
      << ") with respect to mean and std dev (" 
      << mean_2D_noise << "+/-" << sigma_2D_noise << ")!"; 
  }

  // Check second peak is above mean of the noise +/- 2*sigma 
  if ( second_deriv[plateau_edges.second] > mean_2D_noise - 2*sigma_2D_noise ) { 
    LogDebug(mlCommissioning_) 
      << "[VpspScanAnalysis::" << __func__ << "]" 
      << " Noise of second derivative  too large (" 
      << second_deriv[plateau_edges.second] 
      << ") with respect to mean and std dev (" 
      << mean_2D_noise << " +/- " << sigma_2D_noise << ")!"; 
  }

  // Find positions where 2nd deriv peaks flatten
  while ( second_deriv[plateau_edges.first] < 
	  mean_2D_noise - 2*sigma_2D_noise && 
	  plateau_edges.first > 5 ) { plateau_edges.first--; }
  while ( second_deriv[plateau_edges.second] > 
	  mean_2D_noise + 2*sigma_2D_noise && 
	  plateau_edges.first < 55 ) { plateau_edges.second++;}
  
  // Locate optimum VPSP value
  float top_mean = 0., bottom_mean = 0.;
  for ( uint16_t m = 5; m < plateau_edges.first; m++ ) {
    top_mean = ( top_mean*(m-5) + histo->GetBinContent(m) ) / (m-4);
  }
  for ( uint16_t m = plateau_edges.second; m < 56; m++ ) { 
    bottom_mean =
      ( ( bottom_mean * (m-plateau_edges.second) ) + 
	histo->GetBinContent(m) ) / 
      ( m - plateau_edges.second + 1 );
  }
  float optimum = bottom_mean + ( top_mean - bottom_mean ) * fractional_baseline_level;

  // Calc vpsp setting  
  uint16_t vpsp;
  for ( vpsp = plateau_edges.first; 
	vpsp < plateau_edges.second; vpsp++ ) { 
    if ( histo->GetBinContent(vpsp) < optimum ) { 
      optimum = histo->GetBinContent(vpsp);
      break; 
    }
  }
  
  // Set monitorables
  monitorables.clear();
  monitorables.push_back(vpsp); // vpsp setting
  monitorables.push_back(static_cast<uint16_t>(optimum)); // adc level for vpsp setting
  float diff = top_mean - bottom_mean;
  if ( diff <= 0. ) { monitorables.push_back(100); }
  else { monitorables.push_back(static_cast<uint16_t>(100*(optimum-bottom_mean)/diff)); } // fraction 
  monitorables.push_back(static_cast<uint16_t>(plateau_edges.first)); // top plateau edge
  monitorables.push_back(static_cast<uint16_t>(plateau_edges.second)); // bottom plateau edge
  monitorables.push_back(static_cast<uint16_t>(top_mean)); // top plateau mean
  monitorables.push_back(static_cast<uint16_t>(bottom_mean)); // bottom plateau mean
  
}






