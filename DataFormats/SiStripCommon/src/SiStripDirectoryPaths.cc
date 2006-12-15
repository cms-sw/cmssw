#include "DataFormats/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
//
string SiStripHistoNamingScheme::controlPath( const SiStripFecKey::Path& path ) {
  
  stringstream folder; 
  folder << sistrip::root_ << sistrip::dir_ << sistrip::controlView_ << sistrip::dir_;
  if ( path.fecCrate_ != sistrip::invalid_ ) {
    folder << sistrip::fecCrate_ << path.fecCrate_ << sistrip::dir_;
    if ( path.fecSlot_ != sistrip::invalid_ ) {
      folder << sistrip::fecSlot_ << path.fecSlot_ << sistrip::dir_;
      if ( path.fecRing_ != sistrip::invalid_ ) {
	folder << sistrip::fecRing_ << path.fecRing_ << sistrip::dir_;
	if ( path.ccuAddr_ != sistrip::invalid_ ) {
	  folder << sistrip::ccuAddr_ << path.ccuAddr_ << sistrip::dir_;
	  if ( path.ccuChan_ != sistrip::invalid_ ) {
	    folder << sistrip::ccuChan_ << path.ccuChan_ << sistrip::dir_;
	  }
	}
      }
    }
  }
  return folder.str();
}

// -----------------------------------------------------------------------------
//
SiStripFecKey::Path SiStripHistoNamingScheme::controlPath( const string& directory ) {

  SiStripFecKey::Path path;

  uint32_t curr = 0; // current string position
  uint32_t next = 0; // next string position
  next = directory.find( sistrip::controlView_, curr );
  
  // Extract view 
  curr = next;
  if ( curr != string::npos ) { 
    next = directory.find( sistrip::fecCrate_, curr );
    string control_view( directory, 
			 curr+sistrip::controlView_.size(), 
 			 (next-sistrip::dir_.size())-curr );
    
    // Extract FEC crate
    curr = next;
    if ( curr != string::npos ) { 
      next = directory.find( sistrip::fecSlot_, curr );
      string fec_crate( directory, 
			curr+sistrip::fecCrate_.size(), 
			(next-sistrip::dir_.size())-curr );
      path.fecCrate_ = atoi( fec_crate.c_str() );

      // Extract FEC slot
      curr = next;
      if ( curr != string::npos ) { 
	next = directory.find( sistrip::fecRing_, curr );
	string fec_slot( directory, 
			 curr+sistrip::fecSlot_.size(), 
			 (next-sistrip::dir_.size())-curr );
	path.fecSlot_ = atoi( fec_slot.c_str() );

	// Extract FEC ring
	curr = next;
	if ( curr != string::npos ) { 
	  next = directory.find( sistrip::ccuAddr_, curr );
	  string fec_ring( directory, 
			   curr+sistrip::fecRing_.size(),
			   (next-sistrip::dir_.size())-curr );
	  path.fecRing_ = atoi( fec_ring.c_str() );

	  // Extract CCU address
	  curr = next;
	  if ( curr != string::npos ) { 
	    next = directory.find( sistrip::ccuChan_, curr );
	    string ccu_addr( directory, 
			     curr+sistrip::ccuAddr_.size(), 
			     (next-sistrip::dir_.size())-curr );
	    path.ccuAddr_ = atoi( ccu_addr.c_str() );

	    // Extract CCU channel
	    curr = next;
	    if ( curr != string::npos ) { 
	      next = string::npos;
	      string ccu_chan( directory, 
			       curr+sistrip::ccuChan_.size(), 
			       next-curr );
	      path.ccuChan_ = atoi( ccu_chan.c_str() );
	    }
	  }
	}
      }
    }
  } else {
    stringstream ss;
    ss << "[SiStripHistoNamingScheme::" << __func__ << "]" 
       << " Unexpected view: " 
       << SiStripHistoNamingScheme::view( directory )
       << " in directory path: "
       << directory;
    edm::LogWarning(mlDqmCommon_) << ss.str();
  }
  
  return path;
  
}

// -----------------------------------------------------------------------------
//
string SiStripHistoNamingScheme::readoutPath( const SiStripFedKey::Path& path ) { 
  
  stringstream folder;
  folder << sistrip::root_ << sistrip::dir_ << sistrip::readoutView_ << sistrip::dir_;
  if ( path.fedId_ != sistrip::invalid_ ) {
    folder << sistrip::fedId_ << path.fedId_ << sistrip::dir_;
    //if ( path.fedFe_ != sistrip::invalid_ ) {
    //folder << sistrip::feUnit_ << path.fedFe_ << sistrip::dir_;
    //if ( path.feCh_ != sistrip::invalid_ ) {
    //folder << sistrip::fedFeChan_ << path.feCh_ << sistrip::dir_;
    if ( path.fedCh_ != sistrip::invalid_ ) {
      folder << sistrip::fedChannel_ << path.fedCh_ << sistrip::dir_;
    }
  }
  return folder.str();
}

// -----------------------------------------------------------------------------

SiStripFedKey::Path SiStripHistoNamingScheme::readoutPath( const std::string& directory ) { 
  
  SiStripFedKey::Path path;
  
  uint32_t curr = 0; // current string position
  uint32_t next = 0; // next string position
  next = directory.find( sistrip::readoutView_, curr );

  // Extract view 
  curr = next;
  if ( curr != string::npos ) { 
    next = directory.find( sistrip::fedId_, curr );
    string readout_view( directory, 
			 curr+sistrip::readoutView_.size(), 
			 (next-sistrip::dir_.size())-curr );

    // Extract FED id
    curr = next;
    if ( curr != string::npos ) { 
      next = directory.find( sistrip::fedChannel_, curr );
      string fed_id( directory, 
		     curr+sistrip::fedId_.size(), 
		     (next-sistrip::dir_.size())-curr );
      path.fedId_ = atoi( fed_id.c_str() );
      
      // Extract FED channel
      curr = next;
      if ( curr != string::npos ) { 
	next = string::npos;
	string fed_channel( directory, 
			    curr+sistrip::fedChannel_.size(), 
			    next-curr );
	path.fedCh_ = atoi( fed_channel.c_str() );
      }
    }
    
  } else {
    stringstream ss;
    ss << "[SiStripHistoNamingScheme::" << __func__ << "]" 
       << " Unexpected view: " 
       << SiStripHistoNamingScheme::view( directory );
    edm::LogWarning(mlDqmCommon_) << ss.str();
  }

  return path;
  
}
