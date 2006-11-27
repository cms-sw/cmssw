#include "DQM/SiStripCommissioningAnalysis/interface/FedCablingAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "TProfile.h"
#include "TH1.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

using namespace std;

// ----------------------------------------------------------------------------
// 
FedCablingAnalysis::FedCablingAnalysis( const uint32_t& key ) 
  : CommissioningAnalysis(key),
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
  : CommissioningAnalysis(),
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
void FedCablingAnalysis::print( stringstream& ss, uint32_t not_used ) { 
  ss << "[FedCablingAnalysis::" << __func__ << "]";
  if ( key() ) {
    ss << " FED CABLING monitorables for channel key 0x"
       << hex << setw(8) << setfill('0') << key() << dec << endl;
  } else {
    ss << " FED CABLING monitorables" << endl;
  }
  if ( key() ) {
    SiStripFecKey::Path path = SiStripFecKey::path( key() );
    ss << " Crate/FEC/ring/CCU/module/channel: " 
       << path.fecCrate_ << "/"
       << path.fecSlot_ << "/"
       << path.fecRing_ << "/"
       << path.ccuAddr_ << "/"
       << path.ccuChan_ << "/"
       << path.channel_ 
       << endl;
  }
  ss << " nCandidates           : " << candidates_.size() << endl
     << " Candidates (id/ch/adc): ";
  Candidates::const_iterator iter;
  for ( iter = candidates_.begin(); iter != candidates_.end(); iter++ ) { 
    SiStripFedKey::Path path = SiStripFedKey::path( iter->first );
    ss << path.fedId_ << "/" << path.fedCh_ << "/" << iter->second << " ";
  }
  ss << endl
     << " Connected FED id      : " << fedId_ << endl 
     << " Connected FED channel : " << fedCh_ << endl
     << " Signal level [adc]    : " << signalLevel() << endl;
}

// ----------------------------------------------------------------------------
// 
void FedCablingAnalysis::extract( const vector<TH1*>& histos ) { 

  // Check
  if ( histos.size() != 2 ) {
    cerr << endl // edm::LogWarning(mlDqmAnalysis_)
	 << "[FedCablingAnalysis::" << __func__ << "]"
	 << " Unexpected number of histograms: " 
	 << histos.size();
    vector<TH1*>::const_iterator ihis = histos.begin();
    for ( ; ihis != histos.end(); ihis++ ) {
      cout << "[FedCablingAnalysis::" << __func__ << "]"
	   << " Histogram name: " << (*ihis)->GetName() << endl;
    }
  }
  
  // Extract
  vector<TH1*>::const_iterator ihis = histos.begin();
  for ( ; ihis != histos.end(); ihis++ ) {
    
    // Check pointer
    if ( !(*ihis) ) {
      cerr << endl // edm::LogWarning(mlDqmAnalysis_)
	<< "[FedCablingAnalysis::" << __func__ << "]"
	<< " NULL pointer to histogram!";
      continue;
    }

    // Check name
    static HistoTitle title;
    title = SiStripHistoNamingScheme::histoTitle( (*ihis)->GetName() );
    if ( title.task_ != sistrip::FED_CABLING ) {
      cerr << endl // edm::LogWarning(mlDqmAnalysis_)
	<< "[FedCablingAnalysis::" << __func__ << "]"
	<< " Unexpected commissioning task!"
	<< "(" << SiStripHistoNamingScheme::task( title.task_ ) << ")";
      continue;
    }

    // Extract FED id and channel histos
    if ( title.extraInfo_.find(sistrip::fedId_) != string::npos ) {
      hFedId_.first = *ihis;
      hFedId_.second = (*ihis)->GetName();
    } else if ( title.extraInfo_.find(sistrip::fedChannel_) != string::npos ) {
      hFedCh_.first = *ihis;
      hFedCh_.second = (*ihis)->GetName();
    } else { 
      cerr << endl // edm::LogWarning(mlDqmAnalysis_)
	<< "[FedCablingAnalysis::" << __func__ << "]"
	<< " Unexpected 'extra info': " << title.extraInfo_;
    }
    
  }
  
}


// -----------------------------------------------------------------------------
// 
void FedCablingAnalysis::analyse() { 
  
  // Check for valid pointers to histograms
  if ( !hFedId_.first ) {
    cerr << endl // edm::LogWarning(mlDqmAnalysis_)
      << "[FedCablingAnalysis::" << __func__ << "]"
      << " NULL pointer to 'FED id' histogram";
    return;
  }
  if ( !hFedCh_.first ) {
    cerr << endl // edm::LogWarning(mlDqmAnalysis_)
      << "[FedCablingAnalysis::" << __func__ << "]"
      << " NULL pointer to 'FED channel' histogram";
    return;
  }

  TProfile* fedid_histo = dynamic_cast<TProfile*>(hFedId_.first);
  TProfile* fedch_histo = dynamic_cast<TProfile*>(hFedCh_.first);

  if ( !fedid_histo ) {
    cerr << "[" << __PRETTY_FUNCTION__ << "]"
	 << " NULL pointer to FedId TProfile histogram!" << endl;
    return;
  }

  if ( !fedch_histo ) {
    cerr << "[" << __PRETTY_FUNCTION__ << "]"
	 << " NULL pointer to FedCh TProfile histogram!" << endl;
    return;
  }

  // Some initialization
  candidates_.clear();
  float max = -1.*sistrip::invalid_;
  uint16_t id_val = sistrip::invalid_;
  uint16_t ch_val = sistrip::invalid_;

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

  // Set "best" candidate
  fedId_ = id_val;
  fedCh_ = ch_val;
  
}

// -----------------------------------------------------------------------------
//
const uint16_t& FedCablingAnalysis::signalLevel() const { 
  static uint16_t temp = 0; 
  uint32_t key = SiStripFedKey::key( SiStripFedKey::Path(fedId_,fedCh_) );
  Candidates::const_iterator iter = candidates_.find( key );
  if ( iter != candidates_.end() ) { return iter->second; }
  else { return temp; }
}
