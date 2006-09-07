#include "DQM/SiStripCommissioningAnalysis/interface/FedCablingAnalysis.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "DQM/SiStripCommissioningAnalysis/interface/MeanAndStdDev.h"
#include "TProfile.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

using namespace std;

// ----------------------------------------------------------------------------
// 
FedCablingAnalysis::FedCablingAnalysis() 
  : CommissioningAnalysis(),
    fedId_(sistrip::invalid_), 
    fedCh_(sistrip::invalid_),
    level_(sistrip::invalid_),
    num_(sistrip::invalid_),
    hFedId_(0,""),
    hFedCh_(0,"")
{;}

// ----------------------------------------------------------------------------
// 
void FedCablingAnalysis::print( stringstream& ss, uint32_t not_used ) { 
  ss << "FED CABLING Monitorables:" << "\n"
     << " FED id              : " << fedId_ << "\n" 
     << " FED channel         : " << fedCh_ << "\n"
     << " Signal level [adc]  : " << level_ << "\n"
     << " Number of candidates: " << num_ << "\n";
}

// ----------------------------------------------------------------------------
// 
void FedCablingAnalysis::reset() {
    fedId_ = sistrip::invalid_; 
    fedCh_ = sistrip::invalid_;
    level_ = sistrip::invalid_;
    num_   = sistrip::invalid_;
    hFedId_ = Histo(0,"");
    hFedCh_ = Histo(0,"");
}

// ----------------------------------------------------------------------------
// 
void FedCablingAnalysis::extract( const vector<TProfile*>& histos ) { 

  // Check
  if ( histos.size() != 2 ) {
    cerr << "[" << __PRETTY_FUNCTION__ << "]"
	 << " Unexpected number of histograms: " 
	 << histos.size()
	 << endl;
  }
  
  // Extract
  vector<TProfile*>::const_iterator ihis = histos.begin();
  for ( ; ihis != histos.end(); ihis++ ) {
    
    // Check pointer
    if ( !(*ihis) ) {
      cerr << "[" << __PRETTY_FUNCTION__ << "]"
	   << " NULL pointer to histogram!" << endl;
      continue;
    }
    
    // Check name
    static SiStripHistoNamingScheme::HistoTitle title;
    title = SiStripHistoNamingScheme::histoTitle( (*ihis)->GetName() );
    if ( title.task_ != sistrip::FED_CABLING ) {
      cerr << "[" << __PRETTY_FUNCTION__ << "]"
	   << " Unexpected commissioning task!"
	   << "(" << SiStripHistoNamingScheme::task( title.task_ ) << ")"
	   << endl;
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
      cerr << "[" << __PRETTY_FUNCTION__ << "]"
	   << " Unexpected 'extra info': " << title.extraInfo_ << endl;
    }
    
  }
  
}


// -----------------------------------------------------------------------------
// 
void FedCablingAnalysis::analyse() { 
  
  // Check for valid pointers to histograms
  if ( !hFedId_.first ) {
    cerr << "[" << __PRETTY_FUNCTION__ << "]"
	 << " NULL pointer to 'FED id' histogram" << endl;
    return;
  }
  if ( !hFedCh_.first ) {
    cerr << "[" << __PRETTY_FUNCTION__ << "]"
	 << " NULL pointer to 'FED channel' histogram" << endl;
    return;
  }

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
  uint16_t ch_num = 0;
  uint16_t ch_val = sistrip::invalid_;
  float    ch_max = -1.*sistrip::invalid_;
  for ( uint16_t ibin = 0; ibin < hFedCh_.first->GetNbinsX(); ibin++ ) {
    if ( hFedCh_.first->GetBinEntries(ibin+1) ) {
      ch_num++;
      if ( hFedCh_.first->GetBinContent(ibin+1) > ch_max ) {
	ch_max = hFedCh_.first->GetBinContent(ibin+1);
	ch_val = ibin;
      }
    }
  }

  // Set monitorables
  fedId_ = id_val;
  fedCh_ = ch_val;
  level_ = ch_max;
  num_   = ch_num;
  
}
