#include "CondFormats/SiStripObjects/interface/VpspScanAnalysis.h"
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
    histos_( 2, Histo(0,"") )
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
    histos_( 2, Histo(0,"") )
{;}

// -----------------------------------------------------------------------------
//
void VpspScanAnalysis::reset() {
  vpsp_ = VInt(2,sistrip::invalid_);
  histos_.clear();
  histos_.resize( 2, Histo(0,"") );
}

// ----------------------------------------------------------------------------
// 
void VpspScanAnalysis::extract( const std::vector<TH1*>& histos ) { 

  // Check number of histograms
  if ( histos.size() != 2 ) {
    addErrorCode(sistrip::numberOfHistos_);
  }
  
  // Extract FED key from histo title
  if ( !histos.empty() ) extractFedKey( histos.front() );

  // Extract histograms
  std::vector<TH1*>::const_iterator ihis = histos.begin();
  for ( ; ihis != histos.end(); ihis++ ) {
    
    // Check pointer
    if ( !(*ihis) ) { continue; }
    
    // Check name
    SiStripHistoTitle title( (*ihis)->GetName() );
    if ( title.runType() != sistrip::VPSP_SCAN ) {
      addErrorCode(sistrip::unexpectedTask_);
      continue;
    }
    
    // Extract APV number
    uint16_t apv = sistrip::invalid_; 
    if ( title.extraInfo().find(sistrip::apv_) != std::string::npos ) {
      std::stringstream ss;
      ss << title.extraInfo().substr( title.extraInfo().find(sistrip::apv_) + sistrip::apv_.size(), 1 );
      ss >> std::dec >> apv;
    }

    if ( apv <= 1 ) {
      histos_[apv].first = *ihis; 
      histos_[apv].second = (*ihis)->GetName();
    } else {
      addErrorCode(sistrip::unexpectedExtraInfo_);
    }
    
  }

}

// -----------------------------------------------------------------------------
//
void VpspScanAnalysis::analyse() {

  // Use deprecated method
  deprecated(); 
  
}

// ----------------------------------------------------------------------------
// 
bool VpspScanAnalysis::isValid() const {
  return ( vpsp_[0] < 1. * sistrip::valid_ &&
	   vpsp_[1] < 1. * sistrip::valid_ &&
	   adcLevel_[0] < 1. * sistrip::valid_ &&
	   adcLevel_[1] < 1. * sistrip::valid_ &&
	   topLevel_[0] < 1. * sistrip::valid_ &&
	   topLevel_[1] < 1. * sistrip::valid_ &&
	   bottomLevel_[0] < 1. * sistrip::valid_ &&
	   bottomLevel_[1] < 1. * sistrip::valid_ &&
	   getErrorCodes().empty() );
} 

// ----------------------------------------------------------------------------
// 
void VpspScanAnalysis::summary( std::stringstream& ss ) const { 
  
  sistrip::RunType type = SiStripEnumsAndStrings::runType( myName() );
  
  std::stringstream extra1,extra2;
  extra1 << sistrip::apv_ << "0";
  extra2 << sistrip::apv_ << "1";
  
  std::string title1 = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
					  type,
					  sistrip::FED_KEY, 
					  fedKey().key(),
					  sistrip::LLD_CHAN, 
					  fecKey().lldChan(),
					  extra1.str() ).title();
  std::string title2 = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
					  type,
					  sistrip::FED_KEY, 
					  fedKey().key(),
					  sistrip::LLD_CHAN, 
					  fecKey().lldChan(),
					  extra2.str() ).title();
  
  ss << " Summary"
     << ":"
     << ( isValid() ? "Valid" : "Invalid" )
     << ":"
     << sistrip::controlView_ << ":"
     << fecKey().fecCrate() << "/" 
     << fecKey().fecSlot() << "/" 
     << fecKey().fecRing() << "/" 
     << fecKey().ccuAddr() << "/" 
     << fecKey().ccuChan() 
     << ":"
     << sistrip::dqmRoot_ << sistrip::dir_ 
     << "Collate" << sistrip::dir_ 
     << SiStripFecKey( fecKey().fecCrate(),
		       fecKey().fecSlot(), 
		       fecKey().fecRing(), 
		       fecKey().ccuAddr(), 
		       fecKey().ccuChan() ).path()
     << ":"
     << title1 << ";" << title2
     << std::endl;
  
}

// ----------------------------------------------------------------------------
// 
void VpspScanAnalysis::print( std::stringstream& ss, uint32_t iapv ) { 
  if ( iapv == 1 || iapv == 2 ) { iapv--; }
  else { iapv = 0; }
  header( ss );
  ss << " Monitorables for APV : " << iapv;
  if ( iapv == 0 ) { ss << " (first of pair)"; }
  else if ( iapv == 1 ) { ss << " (second of pair)"; } 
  ss << std::endl;
  ss <<  std::fixed << std::setprecision(2)
     << " VPSP setting         : " << vpsp_[iapv] << std::endl 
     << " Signal level   [ADC] : " << adcLevel_[iapv] << std::endl
     << " Fraction         [%] : " << "(N/A)" /*fraction_[iapv]*/ << std::endl
     << " Top edge       [bin] : " << topEdge_[iapv] << std::endl
     << " Bottom edge    [bin] : " << bottomEdge_[iapv] << std::endl
     << " Top level      [ADC] : " << topLevel_[iapv] << std::endl
     << " Bottom level   [ADC] : " << bottomLevel_[iapv] << std::endl
     << std::boolalpha 
     << " isValid              : " << isValid()  << std::endl
     << std::noboolalpha
     << " Error codes (found "  
     << std::setw(2) << std::setfill(' ') << getErrorCodes().size() 
     << "): ";
  if ( getErrorCodes().empty() ) { ss << "(none)"; }
  else { 
    VString::const_iterator istr = getErrorCodes().begin();
    VString::const_iterator jstr = getErrorCodes().end();
    for ( ; istr != jstr; ++istr ) { ss << *istr << " "; }
  }
  ss << std::endl;
}

// -----------------------------------------------------------------------------
//
void VpspScanAnalysis::deprecated() {
  
  std::vector<const TProfile*> histos; 
  std::vector<uint16_t> monitorables;
  for ( uint16_t iapv = 0; iapv < 2; iapv++ ) {
    
    monitorables.clear();
    monitorables.resize( 7, sistrip::invalid_ );

    histos.clear();
    histos.push_back( const_cast<const TProfile*>( dynamic_cast<TProfile*>(histos_[iapv].first) ) );
    
    if ( !histos[0] ) {
      addErrorCode(sistrip::nullPtr_);
      continue;
    }
    
    if ( false ) { anal( histos, monitorables ); }
    else {

      // Find "top" plateau
      int first = sistrip::invalid_;
      float top = -1. * sistrip::invalid_;;
      for ( int k = 5; k < 55; k++ ) {
	if ( !histos[0]->GetBinEntries(k) ) { continue; }
	if ( histos[0]->GetBinContent(k) >= top ) { 
	  first = k; 
	  top = histos[0]->GetBinContent(k); 
	}
      }
      if ( top < -1. * sistrip::valid_ ) { top = sistrip::invalid_; } //@@ just want +ve invalid number here
      if ( top > 1. * sistrip::valid_ ) { 
	addErrorCode(sistrip::noTopPlateau_);
	continue;
      } 
      monitorables[5] = static_cast<uint16_t>(top);
      monitorables[3] = first;
	
      // Find "bottom" plateau
      int last = sistrip::invalid_;
      float bottom = 1. * sistrip::invalid_;
      for ( int k = 55; k > 5; k-- ) {
	if ( !histos[0]->GetBinEntries(k) ) { continue; }
	if ( histos[0]->GetBinContent(k) <= bottom ) { 
	  last = k; 
	  bottom = histos[0]->GetBinContent(k); 
	}
      }
      if ( bottom > 1. * sistrip::valid_ ) {
	addErrorCode(sistrip::noBottomPlateau_);
	continue;
      } 
      monitorables[6] = static_cast<uint16_t>(bottom);
      monitorables[4] = last;
      
      // Set optimum baseline level
      float opt = bottom + ( top - bottom ) * 1./3.; 
      monitorables[1] = static_cast<uint16_t>(opt);
      
      // Find optimum VPSP setting 
      uint16_t vpsp = sistrip::invalid_;
      if ( opt < 1. * sistrip::valid_ ) {
	uint16_t ivpsp = 5; 
	for ( ; ivpsp < 55; ivpsp++ ) { 
	  if ( histos[0]->GetBinContent(ivpsp) < opt ) { break; }
	}
	if ( ivpsp != 54 ) { 
	  vpsp = ivpsp; 
	  monitorables[0] = vpsp;
	}
	else { 
	  addErrorCode(sistrip::noVpspSetting_); 
	  continue;
	}
	
      } else { 
	addErrorCode(sistrip::noBaselineLevel_); 
      	continue;
      }
      
    }
    
    // Set analysis results for both APVs
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






