#include "DQM/SiStripCommissioningSummary/interface/SummaryGeneratorControlView.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumeratedTypes.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include <iostream>
#include <sstream>
#include <cmath>
#include "TProfile.h"
#include "TH1F.h"
#include "TH2F.h"

using namespace std;

// -----------------------------------------------------------------------------
// 
void SummaryGeneratorControlView::fill( const string& top_level_dir,
					const sistrip::Granularity& granularity,
					const uint32_t& device_key, 
					const float& value,
					const float& error ) {
  
  // Check granularity is recognised
  string gran = SiStripHistoNamingScheme::granularity( granularity );

  if ( granularity != sistrip::UNKNOWN_GRAN &&
       granularity != sistrip::FEC_CRATE &&
       granularity != sistrip::FEC_SLOT &&
       granularity != sistrip::FEC_RING &&
       granularity != sistrip::CCU_ADDR &&
       granularity != sistrip::CCU_CHAN &&
       granularity != sistrip::LLD_CHAN && 
       granularity != sistrip::APV ) {
    string temp = SiStripHistoNamingScheme::granularity( sistrip::LLD_CHAN );
    cerr << "[SummaryGeneratorControlView::" << __func__ << "]"
	 << " Unexpected granularity requested: " << gran
	 << endl;
    return;
  }
  
  // Create key representing "top level" directory 
  SiStripFecKey::Path top = SiStripHistoNamingScheme::controlPath( top_level_dir );

  // Path and string for "present working directory" as defined by device key
  SiStripFecKey::Path path = SiStripFecKey::path( device_key );
  string pwd = SiStripHistoNamingScheme::controlPath( path );
  
  // Check path is "within" top-level directory structure 
  if ( ( ( path.fecCrate_ == top.fecCrate_ ) || ( top.fecCrate_ == sistrip::invalid_ ) ) && path.fecCrate_ != sistrip::invalid_ &&
       ( ( path.fecSlot_  == top.fecSlot_  ) || ( top.fecSlot_  == sistrip::invalid_ ) ) && path.fecSlot_  != sistrip::invalid_ &&
       ( ( path.fecRing_  == top.fecRing_  ) || ( top.fecRing_  == sistrip::invalid_ ) ) && path.fecRing_  != sistrip::invalid_ && 
       ( ( path.ccuAddr_  == top.ccuAddr_  ) || ( top.ccuAddr_  == sistrip::invalid_ ) ) && path.ccuAddr_  != sistrip::invalid_ &&
       ( ( path.ccuChan_  == top.ccuChan_  ) || ( top.ccuChan_  == sistrip::invalid_ ) ) && path.ccuChan_  != sistrip::invalid_ ) { 
    
    // Extract path and string corresponding to "top-level down to granularity" 
    string sub_dir = pwd;
    uint32_t pos = pwd.find( gran );
    if ( pos != string::npos ) {
      sub_dir = pwd.substr( 0, pwd.find(sistrip::dir_,pos) );
    } else if ( granularity == sistrip::UNKNOWN_GRAN ) {
      sub_dir = pwd;
    }

    SiStripFecKey::Path sub_path = SiStripHistoNamingScheme::controlPath( sub_dir );
    
    // Construct bin label
    stringstream bin;
    if ( sub_path.fecCrate_ != sistrip::invalid_ ) { bin << sub_path.fecCrate_; }
    if ( sub_path.fecSlot_  != sistrip::invalid_ ) { bin << sistrip::dot_ << sub_path.fecSlot_; }
    if ( sub_path.fecRing_  != sistrip::invalid_ ) { bin << sistrip::dot_ << sub_path.fecRing_; }
    if ( sub_path.ccuAddr_  != sistrip::invalid_ ) { bin << sistrip::dot_ << sub_path.ccuAddr_; }
    if ( sub_path.ccuChan_  != sistrip::invalid_ ) { bin << sistrip::dot_ << sub_path.ccuChan_; }
    if ( ( granularity == sistrip::LLD_CHAN || 
	   granularity == sistrip::APV ) && 
	 path.channel_ != sistrip::invalid_ ) { bin << sistrip::dot_ << path.channel_; }
    
    // Store "value" in appropriate vector within map (key is bin label)
    map_[bin.str()].push_back( Data(value,error) );
    entries_++;

  }
  
}

