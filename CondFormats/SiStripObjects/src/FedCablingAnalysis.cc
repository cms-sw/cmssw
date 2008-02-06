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
const float FedCablingAnalysis::threshold_ = 100.; // [ADC]

// ----------------------------------------------------------------------------
// 
FedCablingAnalysis::FedCablingAnalysis( const uint32_t& key ) 
  : CommissioningAnalysis(key,sistrip::fedCablingAnalysis_),
    fedId_(sistrip::invalid_), 
    fedCh_(sistrip::invalid_),
    adcLevel_(1.*sistrip::invalid_),
    candidates_(),
    hFedId_(0,""),
    hFedCh_(0,"")
{;}

// ----------------------------------------------------------------------------
// 
FedCablingAnalysis::FedCablingAnalysis() 
  : CommissioningAnalysis(sistrip::fedCablingAnalysis_),
    fedId_(sistrip::invalid_), 
    fedCh_(sistrip::invalid_),
    adcLevel_(1.*sistrip::invalid_),
    candidates_(),
    hFedId_(0,""),
    hFedCh_(0,"") 
{;}

// ----------------------------------------------------------------------------
// 
void FedCablingAnalysis::reset() {
    fedId_ = sistrip::invalid_; 
    fedCh_ = sistrip::invalid_;
    adcLevel_ = 1.*sistrip::invalid_;
    candidates_.clear();
    hFedId_ = Histo(0,"");
    hFedCh_ = Histo(0,"");
}

// ----------------------------------------------------------------------------
// 
void FedCablingAnalysis::extract( const std::vector<TH1*>& histos ) { 

  // Check number of histograms
  if ( histos.size() != 2 ) {
    addErrorCode(sistrip::numberOfHistos_);
  }

  // Extract FED key from histo title
  if ( !histos.empty() ) extractFedKey( histos.front() );

  // Extract histograms
  std::vector<TH1*>::const_iterator ihis = histos.begin();
  for ( ; ihis != histos.end(); ihis++ ) {
    
    // Check for NULL pointer
    if ( !(*ihis) ) { continue; }

    // Check name
    SiStripHistoTitle title( (*ihis)->GetName() );
    if ( title.runType() != sistrip::FED_CABLING ) {
      addErrorCode(sistrip::unexpectedTask_);
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
      addErrorCode(sistrip::unexpectedExtraInfo_);
    }
    
  }
  
}


// -----------------------------------------------------------------------------
// 
void FedCablingAnalysis::analyse() { 

  if ( !hFedId_.first ) {
    addErrorCode(sistrip::nullPtr_);
    return;
  }

  if ( !hFedCh_.first ) {
    addErrorCode(sistrip::nullPtr_);
    return;
  }

  TProfile* fedid_histo = dynamic_cast<TProfile*>(hFedId_.first);
  if ( !fedid_histo ) {
    addErrorCode(sistrip::nullPtr_);
    return;
  }

  TProfile* fedch_histo = dynamic_cast<TProfile*>(hFedCh_.first);
  if ( !fedch_histo ) {
    addErrorCode(sistrip::nullPtr_);
    return;
  }

  // Use algorithm to find candidate channels
  algo1( fedid_histo, fedch_histo );
  //algo2( fedid_histo, fedch_histo );
  //algo3( fedid_histo, fedch_histo );

}

// ----------------------------------------------------------------------------
// 
bool FedCablingAnalysis::isValid() const {
  return ( fedId_ < sistrip::maximum_ &&
	   fedCh_ < sistrip::maximum_ &&
	   adcLevel_ < 1+sistrip::maximum_ &&
	   !candidates_.empty() &&
	   getErrorCodes().empty() );
} 

// ----------------------------------------------------------------------------
// 
void FedCablingAnalysis::print( std::stringstream& ss, uint32_t not_used ) { 
  header( ss );
  ss <<  std::fixed << std::setprecision(2)
     << " Connected FED id               : " << fedId_ << std::endl 
     << " Connected FED channel          : " << fedCh_ << std::endl
     << " Signal level             [ADC] : " << adcLevel_ << std::endl;
  ss << " nCandidates                    : " << candidates_.size() << std::endl
     << " FedId/FedChannel/ADC) : ";
  if ( candidates_.empty() ) { ss << "(none)"; }
  else {
    Candidates::const_iterator iter;
    for ( iter = candidates_.begin(); iter != candidates_.end(); iter++ ) { 
      SiStripFedKey path( iter->first );
      ss << path.fedId() << "/" 
	 << path.fedChannel() << "/" 
	 << iter->second << " ";
    }
  }
  ss << std::endl;
  ss << std::boolalpha
     << " isValid                : " << isValid()  << std::endl
     << std::noboolalpha
     << " Error codes (found "  
     << std::setw(2) << std::setfill(' ') << getErrorCodes().size() 
     << ") : ";
  if ( getErrorCodes().empty() ) { ss << "(none)"; }
  else { 
    VString::const_iterator istr = getErrorCodes().begin();
    VString::const_iterator jstr = getErrorCodes().end();
    for ( ; istr != jstr; ++istr ) { ss << *istr << " "; }
  }
  ss << std::endl;
}

// ----------------------------------------------------------------------------
// 
void FedCablingAnalysis::algo1( TProfile* fedid_histo, 
				TProfile* fedch_histo ) { 

  // Some initialization
  candidates_.clear();
  float max       = -1.;
  float weight    = -1.;
  uint16_t id_val = sistrip::invalid_;
  uint16_t ch_val = sistrip::invalid_;
  
  // FED id
  max = 0.;
  for ( uint16_t ifed = 0; ifed < fedid_histo->GetNbinsX(); ifed++ ) {
    if ( fedid_histo->GetBinEntries(ifed+1) ) {
      if ( fedid_histo->GetBinContent(ifed+1) > max &&
	   fedid_histo->GetBinContent(ifed+1) > threshold_ ) { 
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
	   fedch_histo->GetBinContent(ichan+1) > threshold_ ) { 
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
    candidates_[key] = static_cast<uint16_t>(weight);
    fedId_ = id_val;
    fedCh_ = ch_val;
    adcLevel_ = weight;
  } else {
    addErrorCode(sistrip::noCandidates_);
  }

}

// ----------------------------------------------------------------------------
// 
void FedCablingAnalysis::algo2( TProfile* fedid_histo, 
				TProfile* fedch_histo ) { 

  // Some initialization
  candidates_.clear();
  float max       = -1.;
  uint16_t id_val = sistrip::invalid_;
  uint16_t ch_val = sistrip::invalid_;

  // FED id
  for ( uint16_t ifed = 0; ifed < fedid_histo->GetNbinsX(); ifed++ ) {
    if ( fedid_histo->GetBinEntries(ifed+1) ) {

      // FED channel
      for ( uint16_t ichan = 0; ichan < fedch_histo->GetNbinsX(); ichan++ ) {
	if ( fedch_histo->GetBinEntries(ichan+1) ) {

	  // Build FED key
	  uint32_t key = SiStripFedKey( ifed, 
					SiStripFedKey::feUnit(ichan),
					SiStripFedKey::feChan(ichan) ).key();
	  
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
	    adcLevel_ = max; //@@ ok?
	  }

	}
      }
    }
  } 

}

// ----------------------------------------------------------------------------
// 
void FedCablingAnalysis::algo3( TProfile* fedid_histo, 
				TProfile* fedch_histo ) { 

  // Some initialization
  candidates_.clear();
  uint16_t id_num = 0;
  uint16_t id_val = sistrip::invalid_;
  float id_max = -1.;

  // FED id
  for ( uint16_t ibin = 0; ibin < fedid_histo->GetNbinsX(); ibin++ ) {
    if ( fedid_histo->GetBinEntries(ibin+1) ) {
      id_num++;
      if ( fedid_histo->GetBinContent(ibin+1) > id_max ) {
	id_max = fedid_histo->GetBinContent(ibin+1);
	id_val = ibin;
      }
    }
  }

  // FED channel
  uint16_t ch_val = sistrip::invalid_;
  float ch_max = -1.*sistrip::invalid_;
  for ( uint16_t ibin = 0; ibin < fedch_histo->GetNbinsX(); ibin++ ) {
    if ( fedch_histo->GetBinEntries(ibin+1) ) {
      candidates_[ibin] = static_cast<uint16_t>( fedch_histo->GetBinContent(ibin+1) ); 
      if ( candidates_[ibin] > ch_max ) {
	ch_max = candidates_[ibin];
	ch_val = ibin;
      }
    }
  }
  
  // Set monitorables
  fedId_ = id_val;
  fedCh_ = ch_val;
  adcLevel_ = candidates_[ch_val]; //@@ ok?
  
}
