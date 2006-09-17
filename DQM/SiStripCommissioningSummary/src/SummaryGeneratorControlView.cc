#include "DQM/SiStripCommissioningSummary/interface/SummaryGeneratorControlView.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumeratedTypes.h"
#include "DataFormats/SiStripDetId/interface/SiStripControlKey.h"
#include <iostream>
#include <sstream>
#include <cmath>
#include "TProfile.h"
#include "TH1F.h"
#include "TH2F.h"

using namespace std;

// -----------------------------------------------------------------------------
// 
SummaryGeneratorControlView::SummaryGeneratorControlView() {
  // All histos use square of the weights to calc error   
  //TH1::SetDefaultSumw2(true);
}

// -----------------------------------------------------------------------------
// 
void SummaryGeneratorControlView::fill( const string& top_level_dir,
					const sistrip::Granularity& granularity,
					const uint32_t& device_key, 
					const float& value,
					const float& error ) {
  
  // Check granularity is recognised
  string gran = SiStripHistoNamingScheme::granularity( granularity );
  if ( granularity != sistrip::FEC_CRATE &&
       granularity != sistrip::FEC_SLOT &&
       granularity != sistrip::FEC_RING &&
       granularity != sistrip::CCU_ADDR &&
       granularity != sistrip::CCU_CHAN &&
       granularity != sistrip::LLD_CHAN ) { //@@ what about APV?
    cerr << "[" << __PRETTY_FUNCTION__ << "]"
	 << " Unexpected granularity requested: " << gran
	 << endl;
    return;
  }
  
  // Create key representing "top level" directory 
  SiStripHistoNamingScheme::ControlPath top = SiStripHistoNamingScheme::controlPath( top_level_dir );
  
  // Path and string for "present working directory" as defined by device key
  SiStripControlKey::ControlPath path = SiStripControlKey::path( device_key );
  string pwd = SiStripHistoNamingScheme::controlPath( path.fecCrate_,
						      path.fecSlot_,
						      path.fecRing_,
						      path.ccuAddr_,
						      path.ccuChan_ );
  
  // Check path is "within" top-level directory structure 
  if ( ( ( path.fecCrate_ == top.fecCrate_ ) || ( top.fecCrate_ == sistrip::invalid_ ) ) && // && path.fecCrate_ != sistrip::invalid_ ) ) &&
       ( ( path.fecSlot_  == top.fecSlot_  ) || ( top.fecSlot_  == sistrip::invalid_ ) ) && // && path.fecSlot_  != sistrip::invalid_ ) ) &&
       ( ( path.fecRing_  == top.fecRing_  ) || ( top.fecRing_  == sistrip::invalid_ ) ) && // && path.fecRing_  != sistrip::invalid_ ) ) && 
       ( ( path.ccuAddr_  == top.ccuAddr_  ) || ( top.ccuAddr_  == sistrip::invalid_ ) ) && // && path.ccuAddr_  != sistrip::invalid_ ) ) &&
       ( ( path.ccuChan_  == top.ccuChan_  ) || ( top.ccuChan_  == sistrip::invalid_ ) ) ) { // && path.ccuChan_  != sistrip::invalid_ ) ) ) { 

//     // Find top level directory
//     uint32_t pos1 = 0; 
//     if ( top.fecCrate_ == sistrip::invalid_ ) { pos1 = pwd.find( sistrip::fecCrate_ ); }
//     if ( top.fecSlot_  == sistrip::invalid_ ) { pos1 = pwd.find( sistrip::fecSlot_ ); }
//     if ( top.fecRing_  == sistrip::invalid_ ) { pos1 = pwd.find( sistrip::fecRing_ ); }
//     if ( top.ccuAddr_  == sistrip::invalid_ ) { pos1 = pwd.find( sistrip::ccuAddr_ ); }
//     if ( top.ccuChan_  == sistrip::invalid_ ) { pos1 = pwd.find( sistrip::ccuChan_ ); }
//     if ( pos1 == string::npos ) { 
//       cerr << "[" << __PRETTY_FUNCTION__ << "]"
// 	   << " Did not find 'top level directory' within pwd '" << pwd
// 	   << "'!" << endl;
//       return;
//     }

    // Find "granularity" 
    uint32_t pos2 = pwd.find( gran );
    if ( pos2 == string::npos ) { 
      cerr << "[" << __PRETTY_FUNCTION__ << "]"
	   << " Did not find '" << gran
	   << "' within pwd '" << pwd
	   << "'!" << endl;
      return;
    }

    // Extract path and string corresponding to "top-level down to granularity" 
    string sub_dir = pwd.substr( 0/* pos1 */, pwd.find(sistrip::dir_,pos2) );
    SiStripHistoNamingScheme::ControlPath sub_path = SiStripHistoNamingScheme::controlPath( sub_dir );
    
    // Construct bin label
    stringstream bin;
    if ( sub_path.fecCrate_ != sistrip::invalid_ ) { bin << sub_path.fecCrate_; }
    if ( sub_path.fecSlot_  != sistrip::invalid_ ) { bin << sistrip::dot_ << sub_path.fecSlot_; }
    if ( sub_path.fecRing_  != sistrip::invalid_ ) { bin << sistrip::dot_ << sub_path.fecRing_; }
    if ( sub_path.ccuAddr_  != sistrip::invalid_ ) { bin << sistrip::dot_ << sub_path.ccuAddr_; }
    if ( sub_path.ccuChan_  != sistrip::invalid_ ) { bin << sistrip::dot_ << sub_path.ccuChan_; }
    if ( gran == sistrip::LLD_CHAN && 
	 path.channel_  != sistrip::invalid_ ) { bin << sistrip::dot_ << path.channel_; }

//    cout << "[" << __PRETTY_FUNCTION__ << "]" << endl
//         << " top-level: " << top_level_dir << endl
//         << " pwd: " << pwd << endl
//         << " gran: " << gran << endl
//         << " pwd: " << pwd << endl
//         << " pos1: " << pos1 << endl
//         << " pos2: " << pos2 << endl
//         << " sub-dir: "<< sub_dir << endl
//         << " bin nam: " << bin.str()
//         << endl;
    
    // Store "value" in appropriate vector within map (key is bin label)
    map_[bin.str()].push_back( Data(value,error) );
    entries_++;
//     cout << "[" << __PRETTY_FUNCTION__ << "]"
//  	 << " Added value +/- error  " << value << " +/- " << error
//  	 << " to bin with label '" << bin.str()
// 	 << "', which currently has " << map_[bin.str()].size()
// 	 << " entries (map has " << entries_
// 	 << " entries)"
// 	 << endl;
    
  }
  
}

