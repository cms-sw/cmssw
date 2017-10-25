#include "DQM/SiStripCommissioningAnalysis/interface/VpspScanAlgorithm.h"
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
VpspScanAlgorithm::VpspScanAlgorithm( const edm::ParameterSet & pset, VpspScanAnalysis* const anal )
  : CommissioningAlgorithm(anal),
    histos_( 2, Histo(nullptr,"") )
{;}

// ----------------------------------------------------------------------------
// 
void VpspScanAlgorithm::extract( const std::vector<TH1*>& histos ) { 

  if ( !anal() ) {
    edm::LogWarning(mlCommissioning_)
      << "[VpspScanAlgorithm::" << __func__ << "]"
      << " NULL pointer to Analysis object!";
    return; 
  }

  // Check number of histograms
  if ( histos.size() != 2 ) {
    anal()->addErrorCode(sistrip::numberOfHistos_);
  }
  
  // Extract FED key from histo title
  if ( !histos.empty() ) { anal()->fedKey( extractFedKey( histos.front() ) ); }

  // Extract histograms
  std::vector<TH1*>::const_iterator ihis = histos.begin();
  for ( ; ihis != histos.end(); ihis++ ) {
    
    // Check pointer
    if ( !(*ihis) ) { continue; }
    
    // Check name
    SiStripHistoTitle title( (*ihis)->GetName() );
    if ( title.runType() != sistrip::VPSP_SCAN ) {
      anal()->addErrorCode(sistrip::unexpectedTask_);
      continue;
    }
    
    // Extract APV number
    uint16_t apv = sistrip::invalid_; 
    if ( title.extraInfo().find(sistrip::apv_) != std::string::npos ) {
      std::stringstream ss;
      ss << title.extraInfo().substr( title.extraInfo().find(sistrip::apv_) + (sizeof(sistrip::apv_) - 1), 1 );
      ss >> std::dec >> apv;
    }

    if ( apv <= 1 ) {
      histos_[apv].first = *ihis; 
      histos_[apv].second = (*ihis)->GetName();
    } else {
      anal()->addErrorCode(sistrip::unexpectedExtraInfo_);
    }
    
  }

}

// -----------------------------------------------------------------------------
//
void VpspScanAlgorithm::analyse() {

  if ( !anal() ) {
    edm::LogWarning(mlCommissioning_)
      << "[VpspScanAlgorithm::" << __func__ << "]"
      << " NULL pointer to base Analysis object!";
    return; 
  }

  CommissioningAnalysis* tmp = const_cast<CommissioningAnalysis*>( anal() );
  VpspScanAnalysis* anal = dynamic_cast<VpspScanAnalysis*>( tmp );
  if ( !anal ) {
    edm::LogWarning(mlCommissioning_)
      << "[VpspScanAlgorithm::" << __func__ << "]"
      << " NULL pointer to derived Analysis object!";
    return; 
  }

  // from deprecated()
  
  std::vector<const TProfile*> histos; 
  std::vector<uint16_t> monitorables;
  for ( uint16_t iapv = 0; iapv < 2; iapv++ ) {
    
    monitorables.clear();
    monitorables.resize( 7, sistrip::invalid_ );
    
    histos.clear();
    histos.push_back( const_cast<const TProfile*>( dynamic_cast<TProfile*>(histos_[iapv].first) ) );
    
    if ( !histos[0] ) {
      anal->addErrorCode(sistrip::nullPtr_);
      continue;
    }

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
      anal->addErrorCode(sistrip::noTopPlateau_);
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
      anal->addErrorCode(sistrip::noBottomPlateau_);
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
	anal->addErrorCode(sistrip::noVpspSetting_); 
	continue;
      }
	
    } else { 
      anal->addErrorCode(sistrip::noBaselineLevel_); 
      continue;
    }
    
    // Set analysis results for both APVs
    anal->vpsp_[iapv]        = monitorables[0];
    anal->adcLevel_[iapv]    = monitorables[1];
    anal->fraction_[iapv]    = monitorables[2];
    anal->topEdge_[iapv]     = monitorables[3];
    anal->bottomEdge_[iapv]  = monitorables[4];
    anal->topLevel_[iapv]    = monitorables[5];
    anal->bottomLevel_[iapv] = monitorables[6];
    
  }
  
}
