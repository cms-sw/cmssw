#include "DQM/SiStripCommissioningSummary/interface/SummaryGeneratorReadoutView.h"
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
void SummaryGeneratorReadoutView::fill( const string& top_level_dir,
					const sistrip::Granularity& granularity,
					const uint32_t& device_key, 
					const float& value,
					const float& error ) {
  
  // Check granularity is recognised
  string gran = SiStripHistoNamingScheme::granularity( granularity );

  if ( granularity != sistrip::UNKNOWN_GRAN &&
       granularity != sistrip::FED &&
       granularity != sistrip::FED_CHANNEL &&
       granularity != sistrip::FE_UNIT &&
       granularity != sistrip::FE_CHAN &&
       granularity != sistrip::APV ) {
    string temp = SiStripHistoNamingScheme::granularity( sistrip::FE_CHAN );
    cerr << "[SummaryGeneratorReadoutView::" << __func__ << "]"
	 << " Unexpected granularity requested: " << gran
	 << endl;
    return;
  }
  
  // Create key representing "top level" directory 
  SiStripFedKey::Path top = SiStripHistoNamingScheme::readoutPath( top_level_dir );
  
  // Path and string for "present working directory" as defined by device key
  SiStripFedKey::Path path = SiStripFedKey::path( device_key );
  string pwd = SiStripHistoNamingScheme::readoutPath( path );
  
  // Check path is "within" top-level directory structure 
  if ( ( ( path.fedId_ == top.fedId_ ) || ( top.fedId_ == sistrip::invalid_ ) ) && path.fedId_ != sistrip::invalid_ &&
       ( ( path.feUnit_  == top.feUnit_  ) || ( top.feUnit_  == sistrip::invalid_ ) ) && path.feUnit_  != sistrip::invalid_ &&
       ( ( path.feChan_  == top.feChan_  ) || ( top.feChan_  == sistrip::invalid_ ) ) && path.feChan_  != sistrip::invalid_ ) { 
    
    // Extract path and string corresponding to "top-level down to granularity" 
    string sub_dir = pwd;
    uint32_t pos = pwd.find( gran );
    if ( pos != string::npos ) {
      sub_dir = pwd.substr( 0, pwd.find(sistrip::dir_,pos) );
    } else if ( granularity == sistrip::UNKNOWN_GRAN ) {
      sub_dir = pwd;
    }
    
    SiStripFedKey::Path sub_path = SiStripHistoNamingScheme::readoutPath( sub_dir );
    
    // Construct bin label
    stringstream bin;
    if ( sub_path.fedId_ != sistrip::invalid_ ) { bin << sub_path.fedId_; }
    if ( sub_path.feUnit_  != sistrip::invalid_ ) { bin << sistrip::dot_ << sub_path.feUnit_; }
    if ( sub_path.feChan_  != sistrip::invalid_ ) { bin << sistrip::dot_ << sub_path.feChan_; }
    if ( granularity == sistrip::APV &&
	 path.fedApv_ != sistrip::invalid_ ) { bin << sistrip::dot_ << path.fedApv_; }
    
    // Store "value" in appropriate vector within map (key is bin label)
    map_[bin.str()].push_back( Data(value,error) );
    entries_++;
    
  }
  
}

