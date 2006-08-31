#include "DQM/SiStripCommissioningAnalysis/interface/FedCablingAnalysis.h"
#include "DQM/SiStripCommissioningAnalysis/interface/MeanAndStdDev.h"
#include "TProfile.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

#define DBG "FILE: " << __FILE__ << "\n" << "FUNC: " << __PRETTY_FUNCTION__ 

using namespace std;

// -----------------------------------------------------------------------------
// 
void FedCablingAnalysis::analysis( const TProfiles& histos, 
				   FedCablingAnalysis::Monitorables& mons ) { 
  //cout << DBG << endl;
  
  // Check for valid pointers to histograms
  if ( !histos.fedId_ ) {
    cerr << "[" << __PRETTY_FUNCTION__ << "]"
	 << " NULL pointer to 'FED id' histogram" << endl;
    return;
  }
  if ( !histos.fedCh_ ) {
    cerr << "[" << __PRETTY_FUNCTION__ << "]"
	 << " NULL pointer to 'FED channel' histogram" << endl;
    return;
  }

  // FED id
  uint16_t id_num = 0;
  uint16_t id_val = sistrip::invalid_;
  float    id_max = -1.*sistrip::invalid_;
  for ( uint16_t ibin = 0; ibin < histos.fedId_->GetNbinsX(); ibin++ ) {
    if ( histos.fedId_->GetBinEntries(ibin+1) ) {
      id_num++;
      if ( histos.fedId_->GetBinContent(ibin+1) > id_max ) {
	id_max = histos.fedId_->GetBinContent(ibin+1);
	id_val = ibin;
      }
    }
  }

  // FED channel
  uint16_t ch_num = 0;
  uint16_t ch_val = sistrip::invalid_;
  float    ch_max = -1.*sistrip::invalid_;
  for ( uint16_t ibin = 0; ibin < histos.fedCh_->GetNbinsX(); ibin++ ) {
    if ( histos.fedCh_->GetBinEntries(ibin+1) ) {
      ch_num++;
      if ( histos.fedCh_->GetBinContent(ibin+1) > ch_max ) {
	ch_max = histos.fedCh_->GetBinContent(ibin+1);
	ch_val = ibin;
      }
    }
  }

  // Set monitorables
  mons.fedId_ = id_val;
  mons.fedCh_ = ch_val;
  mons.level_ = ch_max;
  mons.num_   = ch_num;
  
}

// ----------------------------------------------------------------------------
// 
void FedCablingAnalysis::Monitorables::print( stringstream& ss ) { 
  ss << "FED CABLING Monitorables:" << "\n"
     << " FED id              : " << fedId_ << "\n" 
     << " FED channel         : " << fedCh_ << "\n"
     << " Signal level [adc]  : " << level_ << "\n"
     << " Number of candidates: " << num_ << "\n";
}
