#include "DQM/SiStripCommissioningAnalysis/interface/FedCablingAlgorithm.h"
#include "CondFormats/SiStripObjects/interface/FedCablingAnalysis.h"
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

// ----------------------------------------------------------------------------
// 
FedCablingAlgorithm::FedCablingAlgorithm( const edm::ParameterSet & pset, FedCablingAnalysis* const anal ) 
  : CommissioningAlgorithm(anal),
    hFedId_(nullptr,""),
    hFedCh_(nullptr,"")
{;}

// ----------------------------------------------------------------------------
// 
void FedCablingAlgorithm::extract( const std::vector<TH1*>& histos ) { 

  if ( !anal() ) {
    edm::LogWarning(mlCommissioning_)
      << "[FedCablingAlgorithm::" << __func__ << "]"
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
    
    // Check for NULL pointer
    if ( !(*ihis) ) { continue; }

    // Check name
    SiStripHistoTitle title( (*ihis)->GetName() );
    if ( title.runType() != sistrip::FED_CABLING ) {
      anal()->addErrorCode(sistrip::unexpectedTask_);
      continue;
    }
    
    // Extract FED id and channel histos
    if ( title.extraInfo().find(sistrip::feDriver_) != std::string::npos ) {
      hFedId_.first = *ihis;
      hFedId_.second = (*ihis)->GetName();
    } else if ( title.extraInfo().find(sistrip::fedChannel_) != std::string::npos ) {
      hFedCh_.first = *ihis;
      hFedCh_.second = (*ihis)->GetName();
    } else { 
      anal()->addErrorCode(sistrip::unexpectedExtraInfo_);
    }
    
  }
  
}

// -----------------------------------------------------------------------------
// 
void FedCablingAlgorithm::analyse() { 

  if ( !anal() ) {
    edm::LogWarning(mlCommissioning_)
      << "[FedCablingAlgorithm::" << __func__ << "]"
      << " NULL pointer to base Analysis object!";
    return; 
  }

  CommissioningAnalysis* tmp = const_cast<CommissioningAnalysis*>( anal() );
  FedCablingAnalysis* anal = dynamic_cast<FedCablingAnalysis*>( tmp );
  if ( !anal ) {
    edm::LogWarning(mlCommissioning_)
      << "[FedCablingAlgorithm::" << __func__ << "]"
      << " NULL pointer to derived Analysis object!";
    return; 
  }

  if ( !hFedId_.first ) {
    anal->addErrorCode(sistrip::nullPtr_);
    return;
  }

  if ( !hFedCh_.first ) {
    anal->addErrorCode(sistrip::nullPtr_);
    return;
  }

  TProfile* fedid_histo = dynamic_cast<TProfile*>(hFedId_.first);
  if ( !fedid_histo ) {
    anal->addErrorCode(sistrip::nullPtr_);
    return;
  }

  TProfile* fedch_histo = dynamic_cast<TProfile*>(hFedCh_.first);
  if ( !fedch_histo ) {
    anal->addErrorCode(sistrip::nullPtr_);
    return;
  }

  // Some initialization
  anal->candidates_.clear();
  float max       = -1.;
  float weight    = -1.;
  uint16_t id_val = sistrip::invalid_;
  uint16_t ch_val = sistrip::invalid_;
  
  // FED id
  max = 0.;
  for ( uint16_t ifed = 0; ifed < fedid_histo->GetNbinsX(); ifed++ ) {
    if ( fedid_histo->GetBinEntries(ifed+1) ) {
      if ( fedid_histo->GetBinContent(ifed+1) > max &&
	   fedid_histo->GetBinContent(ifed+1) > FedCablingAnalysis::threshold_ ) { 
	id_val = ifed; 
	max = fedid_histo->GetBinContent(ifed+1);
      }
    }
  }
  weight = max;

  // FED ch
  max = 0.;
  for ( uint16_t ichan = 0; ichan < fedch_histo->GetNbinsX(); ichan++ ) {
    if ( fedch_histo->GetBinEntries(ichan+1) ) {
      if ( fedch_histo->GetBinContent(ichan+1) > max &&
	   fedch_histo->GetBinContent(ichan+1) > FedCablingAnalysis::threshold_ ) { 
	ch_val = ichan; 
	max = fedch_histo->GetBinContent(ichan+1);
      }
    }
  }
  if ( max > weight ) { weight = max; }

  // Set "best" candidate and ADC level
  if  ( id_val != sistrip::invalid_ &&
	ch_val != sistrip::invalid_ ) {
    uint32_t key = SiStripFedKey( id_val, 
				  SiStripFedKey::feUnit(ch_val),
				  SiStripFedKey::feChan(ch_val) ).key();
    anal->candidates_[key] = static_cast<uint16_t>(weight);
    anal->fedId_ = id_val;
    anal->fedCh_ = ch_val;
    anal->adcLevel_ = weight;
  } else {
    anal->addErrorCode(sistrip::noCandidates_);
  }

}
