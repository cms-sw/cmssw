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

  // Transfer "FED id" histogram contents to containers
  MeanAndStdDev mean_id;
  vector<float> fedid_contents(0);
  vector<float> fedid_errors(0);
  vector<float> fedid_entries(0);
  for ( uint16_t ibin = 0; ibin < histos.fedId_->GetNbinsX(); ibin++ ) {
    fedid_contents.push_back( histos.fedId_->GetBinContent(ibin+1) );
    fedid_errors.push_back( histos.fedId_->GetBinError(ibin+1) );
    fedid_entries.push_back( histos.fedId_->GetBinEntries(ibin+1) );
    mean_id.add( fedid_contents[ibin], fedid_errors[ibin] );
  }
  MeanAndStdDev::Params params_id;
  mean_id.fit( params_id );

  // Transfer "FED channel" histogram contents to containers
  MeanAndStdDev mean_ch;
  vector<float> fedch_contents(0);
  vector<float> fedch_errors(0);
  vector<float> fedch_entries(0);
  for ( uint16_t ibin = 0; ibin < histos.fedCh_->GetNbinsX(); ibin++ ) {
    fedch_contents.push_back( histos.fedCh_->GetBinContent(ibin+1) );
    fedch_errors.push_back( histos.fedCh_->GetBinError(ibin+1) );
    fedch_entries.push_back( histos.fedCh_->GetBinEntries(ibin+1) );
    mean_ch.add( fedch_contents[ibin], fedch_errors[ibin] );
  }
  MeanAndStdDev::Params params_ch;
  mean_ch.fit( params_ch );
  
  cout << "FED id:"
       << " mean: " << params_id.mean_
       << " rms: " << params_id.rms_
       << " median: " << params_id.median_
       << endl;
  cout << "FED ch:"
       << " mean: " << params_ch.mean_
       << " rms: " << params_ch.rms_
       << " median: " << params_ch.median_
       << endl;

  // Identify FED ids with "signal" 
  vector<uint16_t> fed_ids;
  for ( uint16_t ibin = 0; ibin < fedid_contents.size(); ibin++ ) {
    if ( fedid_contents[ibin] > params_id.median_ + 5.*params_id.rms_ ) { //@@ use mean or median?
      fed_ids.push_back(ibin);
    }
  }
  
  // Identify FED ids with "signal" 
  vector<uint16_t> fed_chs;
  for ( uint16_t ibin = 0; ibin < fedch_contents.size(); ibin++ ) {
    if ( fedch_contents[ibin] > params_ch.median_ + 5.*params_ch.rms_ ) { //@@ use mean or median?
      fed_chs.push_back(ibin);
    }
  }
  
  // Check on number of FED ids found
  // if ( fed_chs.size() != 1 ) {
  { 
    stringstream ss;
    ss << "[" << __PRETTY_FUNCTION__ << "]"
       << " Unexpected numher of FED ids identified ("
       << fed_chs.size() << " in total): ";
    for ( uint16_t ifed = 0; ifed < fed_chs.size(); ifed++ ) { 
      ss << fed_chs[ifed] << " ";
    }
    ss << endl;
    cout << ss.str();
  }
  
  // Check on number of FED channels found
  // if ( fed_chs.size() != 1 ) {
  {
    stringstream ss;
    ss << "[" << __PRETTY_FUNCTION__ << "]"
       << " Unexpected numher of FED ids identified ("
       << fed_chs.size() << " in total): ";
    for ( uint16_t ifed = 0; ifed < fed_chs.size(); ifed++ ) { 
      ss << fed_chs[ifed] << " ";
    }
    ss << endl;
    cout << ss.str();
  }

  // Record FED id and channel
  if ( fed_ids.empty() ) { mons.fedId_ = fed_ids[0]; }
  if ( fed_chs.empty() ) { mons.fedCh_ = fed_chs[0]; }
  
}

// ----------------------------------------------------------------------------
// 
void FedCablingAnalysis::Monitorables::print( stringstream& ss ) { 
  ss << "FED CABLING Monitorables:" << "\n"
     << " FED id     : " << fedId_ << "\n" 
     << " FED channel: " << fedCh_ << "\n";
}
