#include "DQM/SiStripCommissioningAnalysis/interface/FedCablingAnalysis.h"
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
FedCablingAnalysis::FedCablingAnalysis( const uint32_t& key ) 
  : CommissioningAnalysis(key,"FedCablingAnalysis"),
    fedId_(sistrip::invalid_), 
    fedCh_(sistrip::invalid_),
    candidates_(),
    hFedId_(0,""),
    hFedCh_(0,"")
{
  reset(); 
}

// ----------------------------------------------------------------------------
// 
FedCablingAnalysis::FedCablingAnalysis() 
  : CommissioningAnalysis("FedCablingAnalysis"),
    fedId_(sistrip::invalid_), 
    fedCh_(sistrip::invalid_),
    candidates_(),
    hFedId_(0,""),
    hFedCh_(0,"") 
{ 
  reset(); 
}

// ----------------------------------------------------------------------------
// 
void FedCablingAnalysis::reset() {
    fedId_ = sistrip::invalid_; 
    fedCh_ = sistrip::invalid_;
    candidates_.clear();
    hFedId_ = Histo(0,"");
    hFedCh_ = Histo(0,"");
}

// ----------------------------------------------------------------------------
// 
void FedCablingAnalysis::extract( const std::vector<TH1*>& histos ) { 

  // Check
  if ( histos.size() != 2 ) {
    edm::LogWarning(mlCommissioning_)
      << "[" << myName() << "::" << __func__ << "]"
      << " Unexpected number of histograms: " 
      << histos.size();
    std::vector<TH1*>::const_iterator ihis = histos.begin();
    for ( ; ihis != histos.end(); ihis++ ) {
      LogTrace(mlCommissioning_)
	<< "[" << myName() << "::" << __func__ << "]"
	<< " Histogram name: " << (*ihis)->GetName();
    }
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
    if ( title.runType() != sistrip::FED_CABLING ) {
      edm::LogWarning(mlCommissioning_)
	<< "[" << myName() << "::" << __func__ << "]"
	<< " Unexpected commissioning task: "
	<< SiStripEnumsAndStrings::runType(title.runType());
      continue;
    }
    
    // Extract FED id and channel histos
    if ( title.extraInfo().find(sistrip::fedId_) != std::string::npos ) {
      hFedId_.first = *ihis;
      hFedId_.second = (*ihis)->GetName();
    } else if ( title.extraInfo().find(sistrip::fedChannel_) != std::string::npos ) {
      hFedCh_.first = *ihis;
      hFedCh_.second = (*ihis)->GetName();
    } else { 
      edm::LogWarning(mlCommissioning_)
	<< "[" << myName() << "::" << __func__ << "]"
	<< " Unexpected 'extra info': " << title.extraInfo();
    }
    
  }
  
}


// -----------------------------------------------------------------------------
// 
void FedCablingAnalysis::analyse() { 
  
  // Check for valid pointers to histograms
  if ( !hFedId_.first ) {
    edm::LogWarning(mlCommissioning_)
      << "[" << myName() << "::" << __func__ << "]"
      << " NULL pointer to 'FED id' histogram";
    return;
  }
  if ( !hFedCh_.first ) {
    edm::LogWarning(mlCommissioning_)
      << "[" << myName() << "::" << __func__ << "]"
      << " NULL pointer to 'FED channel' histogram";
    return;
  }

  TProfile* fedid_histo = dynamic_cast<TProfile*>(hFedId_.first);
  TProfile* fedch_histo = dynamic_cast<TProfile*>(hFedCh_.first);

  if ( !fedid_histo ) {
    edm::LogWarning(mlCommissioning_) 
      << "[" << myName() << "::" << __func__ << "]"
      << " NULL pointer to FedId TProfile histogram!";
    return;
  }

  if ( !fedch_histo ) {
    edm::LogWarning(mlCommissioning_) 
      << "[" << myName() << "::" << __func__ << "]"
      << " NULL pointer to FedCh TProfile histogram!";
    return;
  }

  // Some initialization
  candidates_.clear();
  float max       = sistrip::invalid_ * -1.;
  float weight    = sistrip::invalid_ * -1.;
  uint16_t id_val = sistrip::invalid_;
  uint16_t ch_val = sistrip::invalid_;
  
  // FED id
  max = 0.;
  for ( uint16_t ifed = 0; ifed < fedid_histo->GetNbinsX(); ifed++ ) {
    if ( fedid_histo->GetBinEntries(ifed+1) ) {
      if ( fedid_histo->GetBinContent(ifed+1) > max &&
	   fedid_histo->GetBinContent(ifed+1) > 100. ) { 
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
	   fedch_histo->GetBinContent(ichan+1) > 100. ) { 
	ch_val = ichan; 
	max = fedch_histo->GetBinContent(ichan+1);
      }
    }
  }
  if ( max > weight ) { weight = max; }

  if  ( id_val != sistrip::invalid_ &&
	ch_val != sistrip::invalid_ ) {
    // Set "best" candidate and ADC level
    uint32_t key = SiStripFedKey( id_val, 
				  SiStripFedKey::feUnit(ch_val),
				  SiStripFedKey::feChan(ch_val) ).key();
    candidates_[key] = static_cast<uint16_t>(weight);
    fedId_ = id_val;
    fedCh_ = ch_val;
  }
  
}

// -----------------------------------------------------------------------------
//
uint16_t FedCablingAnalysis::adcLevel() const { 
  uint32_t key = SiStripFedKey( fedId_,
				SiStripFedKey::feUnit(fedCh_),
				SiStripFedKey::feChan(fedCh_) ).key();
  Candidates::const_iterator iter = candidates_.find( key );
  if ( iter != candidates_.end() ) { return iter->second; }
  else { return 0; }
}

// ----------------------------------------------------------------------------
// 
bool FedCablingAnalysis::isValid() {
  return ( fedId_ < sistrip::maximum_ &&
	   fedCh_ < sistrip::maximum_ );
} 

// ----------------------------------------------------------------------------
// 
void FedCablingAnalysis::print( std::stringstream& ss, uint32_t not_used ) { 
  header( ss );
  ss << " nCandidates           : " << candidates_.size() << std::endl
     << " Candidates (id/ch/adc): ";
  Candidates::const_iterator iter;
  for ( iter = candidates_.begin(); iter != candidates_.end(); iter++ ) { 
    SiStripFedKey path( iter->first );
    ss << path.fedId() << "/" 
       << path.fedChannel() << "/" 
       << iter->second << " ";
  }
  ss << std::endl
     << " Connected FED id      : " << fedId_ << std::endl 
     << " Connected FED channel : " << fedCh_ << std::endl
     << " Signal level [adc]    : " << adcLevel();
}










/* ORIGINAL ALGORITHM

// FED id
for ( uint16_t ifed = 0; ifed < fedid_histo->GetNbinsX(); ifed++ ) {
if ( fedid_histo->GetBinEntries(ifed+1) ) {
// FED channel
for ( uint16_t ichan = 0; ichan < fedch_histo->GetNbinsX(); ichan++ ) {
if ( fedch_histo->GetBinEntries(ichan+1) ) {
// Build FED key
SiStripFedKey::Path path( ifed, ichan );
uint32_t key = SiStripFedKey::key( path );
// Calc weighted bin contents from FED id and ch histos
float weight = 
fedid_histo->GetBinContent(ifed+1) * fedid_histo->GetBinEntries(ifed+1) + 
fedch_histo->GetBinContent(ichan+1) * fedch_histo->GetBinEntries(ichan+1);
weight /= ( fedid_histo->GetBinEntries(ifed+1) + fedch_histo->GetBinEntries(ichan+1) );
// Record candidates and "best" candidate
candidates_[key] = static_cast<uint16_t>(weight);
if ( candidates_[key] > max ) {
max = candidates_[key];
id_val = ifed;
ch_val = ichan;
}
}
}
}
} 

*/

/* ANOTHER ALGO

// FED id
uint16_t id_num = 0;
uint16_t id_val = sistrip::invalid_;
float    id_max = -1.*sistrip::invalid_;
for ( uint16_t ibin = 0; ibin < hFedId_.first->GetNbinsX(); ibin++ ) {
if ( hFedId_.first->GetBinEntries(ibin+1) ) {
id_num++;
if ( hFedId_.first->GetBinContent(ibin+1) > id_max ) {
id_max = hFedId_.first->GetBinContent(ibin+1);
id_val = ibin;
}
}
}

// FED channel
std::map<uint16_t,uin16_t> candidates;
uint16_t ch_val = sistrip::invalid_;
float ch_max = -1.*sistrip::invalid_;
for ( uint16_t ibin = 0; ibin < hFedCh_.first->GetNbinsX(); ibin++ ) {
if ( hFedCh_.first->GetBinEntries(ibin+1) ) {
candidates[ibin] = hFedCh_.first->GetBinContent(ibin+1); 
if ( candidates[ibin] > ch_max ) {
ch_max = candidates[ibin];
ch_val = ibin;
}
}
}
  
// Set monitorables
fedId_ = id_val;
fedCh_ = ch_val;
candidates_ = candidates;

*/
