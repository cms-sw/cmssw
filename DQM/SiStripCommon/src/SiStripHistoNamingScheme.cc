#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>
#include <iomanip>

// -----------------------------------------------------------------------------
// definition of static data members
const string SiStripHistoNamingScheme::root_ = "/";
const string SiStripHistoNamingScheme::top_ = "SiStrip";
const string SiStripHistoNamingScheme::dir_ = "/";
const string SiStripHistoNamingScheme::sep_ = "_";
const uint16_t SiStripHistoNamingScheme::all_ = 0xFFFF;

const string SiStripHistoNamingScheme::controlView_ = "ControlView";
const string SiStripHistoNamingScheme::fecCrate_ = "FecCrate";
const string SiStripHistoNamingScheme::fecSlot_ = "FecSlot";
const string SiStripHistoNamingScheme::fecRing_ = "FecRing";
const string SiStripHistoNamingScheme::ccuAddr_ = "CcuAddr";
const string SiStripHistoNamingScheme::ccuChan_ = "CcuChan";

const string SiStripHistoNamingScheme::readoutView_ = "ReadoutView";
const string SiStripHistoNamingScheme::fedId_ = "FedId";
const string SiStripHistoNamingScheme::fedChannel_ = "FedChannel";

const string SiStripHistoNamingScheme::detectorView_ = "DetectorView"; //@@ necessary?

const string SiStripHistoNamingScheme::fecKey_ = "FecKey";
const string SiStripHistoNamingScheme::fedKey_ = "FedKey";
const string SiStripHistoNamingScheme::detKey_ = "DetId"; //@@ necessary?
const string SiStripHistoNamingScheme::unknownKey_ = "UnknownKey";

const string SiStripHistoNamingScheme::sum2_ = "SumOfSquares";
const string SiStripHistoNamingScheme::sum_ = "SumOfContents";
const string SiStripHistoNamingScheme::num_ = "NumOfEntries";
const string SiStripHistoNamingScheme::unknownType_ = "UnknownType";

const string SiStripHistoNamingScheme::lldChan_ = "LldChan";
const string SiStripHistoNamingScheme::apvPair_ = "ApvPair";
const string SiStripHistoNamingScheme::apv_ = "Apv";
const string SiStripHistoNamingScheme::unknownGranularity_ = "UnknownGranularity";

// -----------------------------------------------------------------------------
//
string SiStripHistoNamingScheme::controlPath( uint16_t fec_crate,
					      uint16_t fec_slot,
					      uint16_t fec_ring,
					      uint16_t ccu_addr,
					      uint16_t ccu_chan ) { 
  stringstream folder;
  folder << controlView_;
  if ( fec_crate != all_ ) {
    folder << dir_ << fecCrate_ << fec_crate;
    if ( fec_slot != all_ ) {
      folder << dir_ << fecSlot_ << fec_slot;
      if ( fec_ring != all_ ) {
	folder << dir_ << fecRing_ << fec_ring;
	if ( ccu_addr != all_ ) {
	  folder << dir_ << ccuAddr_ << ccu_addr;
	  if ( ccu_chan != all_ ) {
	    folder << dir_ << ccuChan_ << ccu_chan;
	  }
	}
      }
    }
  }
  LogDebug("DQM") << "[SiStripHistoNamingScheme::controlPath]" << folder.str();
  return folder.str();
}

// -----------------------------------------------------------------------------
//
SiStripHistoNamingScheme::ControlPath SiStripHistoNamingScheme::controlPath( string directory ) {

  ControlPath path;
  path.fecCrate_ = all_;
  path.fecSlot_ = all_;
  path.fecRing_ = all_;
  path.ccuAddr_ = all_;
  path.ccuChan_ = all_;

  uint16_t index = 0;
  
  // Extract view 
  
  stringstream ss; ss << controlView_ << dir_;
  uint16_t size = controlView_.size() + dir_.size();
  edm::LogInfo("compare") << ss.str() << " " << size << " " << directory.compare( index, size, ss.str() );
  if ( !directory.compare( index, size, ss.str() ) ) {
    unsigned short index = controlView_.size() + dir_.size();
    // Extract FEC crate
    if ( !directory.compare( index, fecCrate_.size(), fecCrate_ ) ) {
      index += fecCrate_.size();
      string fec_crate( directory, index, directory.find( directory, index ) - index );
      path.fecCrate_ = atoi( fec_crate.c_str() );
      index = directory.find( dir_, index ) + 1;
      // Extract FEC slot
      if ( !directory.compare( index, fecSlot_.size(), fecSlot_ ) ) {
	index += fecSlot_.size();
	string fec_slot( directory, index, directory.find( directory, index ) - index );
	path.fecSlot_ = atoi( fec_slot.c_str() );
	index = directory.find( dir_, index ) + 1;
	// Extract FEC ring
	if ( !directory.compare( index, fecRing_.size(), fecRing_ ) ) {
	  index += fecRing_.size();
	  string fec_ring( directory, index, directory.find( directory, index ) - index );
	  path.fecRing_ = atoi( fec_ring.c_str() );
	  index = directory.find( dir_, index ) + 1;
	  // Extract CCU address
	  if ( !directory.compare( index, ccuAddr_.size(), ccuAddr_ ) ) {
	    index += ccuAddr_.size();
	    string ccu_addr( directory, index, directory.find( directory, index ) - index );
	    path.ccuAddr_ = atoi( ccu_addr.c_str() );
	    index = directory.find( dir_, index ) + 1;
	    // Extract CCU channel
	    if ( !directory.compare( index, ccuChan_.size(), ccuChan_ ) ) {
	      index += ccuChan_.size();
	      string ccu_chan( directory, index, directory.find( directory, index ) - index );
	      path.ccuChan_ = atoi( ccu_chan.c_str() );
	      index = directory.find( dir_, index ) + 1;
	    }
	  }
	}
      }
    }
  } else {
    edm::LogError("DQM") << "[SiStripHistoNamingScheme::controlPath]" 
			 << " Unexpected view! Not " << controlView_ << "!";
  }
 
  LogDebug("DQM") << "[SiStripHistoNamingScheme::controlPath]" 
		  << "  FecCrate: " << path.fecCrate_
		  << "  FecSlot: " << path.fecSlot_
		  << "  FecRing: " << path.fecRing_
		  << "  CcuAddr: " << path.ccuAddr_
		  << "  CcuChan: " << path.ccuChan_;
  return path;
  
}

// -----------------------------------------------------------------------------
//
string SiStripHistoNamingScheme::readoutPath( uint16_t fed_id,
					      uint16_t fed_channel ) { 
  
  stringstream folder;
  folder << readoutView_;
  if ( fed_id != all_ ) {
    folder << dir_ << fedId_ << fed_id;
    if ( fed_channel != all_ ) {
      folder << dir_ << fedChannel_ << fed_channel;
    }
  }
  return folder.str();
}

// -----------------------------------------------------------------------------
// 
string SiStripHistoNamingScheme::histoName( string      his_name, 
					    HistoType   his_type,
					    KeyType     key_type,
					    uint32_t    his_key,
					    Granularity granularity,
					    uint16_t    channel ) {
  
  stringstream name;
  name << his_name;
  
  stringstream type;
  if      ( his_type == SiStripHistoNamingScheme::SUM2 )    { type << sep_ << sum2_; }
  else if ( his_type == SiStripHistoNamingScheme::SUM )     { type << sep_ << sum_; }
  else if ( his_type == SiStripHistoNamingScheme::NUM )     { type << sep_ << num_; }
  else if ( his_type == SiStripHistoNamingScheme::NO_TYPE ) { /* add nothing */ }
  else { edm::LogError("DQM") << "[SiStripHistoNamingScheme::histoName]"
			      << " Unexpected histogram type!"; }
  name << type.str();
  
  stringstream key;
  if      ( key_type == SiStripHistoNamingScheme::FED )    { key << sep_ << fedKey_ << setfill('0') << setw(8) << hex << his_key; }
  else if ( key_type == SiStripHistoNamingScheme::FEC )    { key << sep_ << fecKey_ << setfill('0') << setw(8) << hex << his_key; }
  else if ( key_type == SiStripHistoNamingScheme::DET )    { key << sep_ << detKey_ << setfill('0') << setw(8) << hex << his_key; }
  else if ( key_type == SiStripHistoNamingScheme::NO_KEY ) { /* add nothing */ }
  else { edm::LogError("DQM") << "[SiStripHistoNamingScheme::histoName]"
			      << " Unexpected key type!"; }
  name << key.str();

  stringstream gran;
  if      ( granularity == SiStripHistoNamingScheme::LLD_CHAN ) { gran << sep_ << lldChan_ << channel; }
  else if ( granularity == SiStripHistoNamingScheme::APV_PAIR ) { gran << sep_ << apvPair_ << channel; }
  else if ( granularity == SiStripHistoNamingScheme::APV )      { gran << sep_ << apv_     << channel; }
  else if ( granularity == SiStripHistoNamingScheme::MODULE )   { /* add nothing */ }
  else { edm::LogError("DQM") << "[SiStripHistoNamingScheme::histoName]"
			      << " Unexpected granularity!"; }
  name << gran.str();
  
  LogDebug("DQM") << "[SiStripHistoNamingScheme::histoName]" << name.str();
  return name.str();
  
}

// -----------------------------------------------------------------------------
// 
SiStripHistoNamingScheme::HistoName SiStripHistoNamingScheme::histoName( string histo_name ) {
  
  HistoName name;
  name.histoName_   = "";
  name.histoType_   = SiStripHistoNamingScheme::UNKNOWN_TYPE;
  name.keyType_     = SiStripHistoNamingScheme::UNKNOWN_KEY;
  name.histoKey_    = 0;
  name.granularity_ = SiStripHistoNamingScheme::UNKNOWN_GRAN;
  name.channel_     = 0;

  // code here.

  return name;
  
}
