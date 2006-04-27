#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>
#include <iomanip>

// -----------------------------------------------------------------------------
//
string SiStripHistoNamingScheme::controlPath( uint16_t fec_crate,
					      uint16_t fec_slot,
					      uint16_t fec_ring,
					      uint16_t ccu_addr,
					      uint16_t ccu_chan ) { 
  stringstream folder;
  folder << sistrip::controlView_;
  if ( fec_crate != sistrip::all_ ) {
    folder << sistrip::dir_ << sistrip::fecCrate_ << fec_crate;
    if ( fec_slot != sistrip::all_ ) {
      folder << sistrip::dir_ << sistrip::fecSlot_ << fec_slot;
      if ( fec_ring != sistrip::all_ ) {
	folder << sistrip::dir_ << sistrip::fecRing_ << fec_ring;
	if ( ccu_addr != sistrip::all_ ) {
	  folder << sistrip::dir_ << sistrip::ccuAddr_ << ccu_addr;
	  if ( ccu_chan != sistrip::all_ ) {
	    folder << sistrip::dir_ << sistrip::ccuChan_ << ccu_chan;
	  }
	}
      }
    }
  }
  LogDebug("DQM") << "[SiStripHistoNamingScheme::controlPath]  " << folder.str();
  return folder.str();
}

// -----------------------------------------------------------------------------
//
SiStripHistoNamingScheme::ControlPath SiStripHistoNamingScheme::controlPath( string directory ) {

  ControlPath path;
  path.fecCrate_ = sistrip::all_;
  path.fecSlot_ = sistrip::all_;
  path.fecRing_ = sistrip::all_;
  path.ccuAddr_ = sistrip::all_;
  path.ccuChan_ = sistrip::all_;

  uint16_t index = 0;
  
  // Extract view 
  stringstream ss; ss << sistrip::controlView_ << sistrip::dir_;
  uint16_t size = sistrip::controlView_.size() + sistrip::dir_.size();
  if ( !directory.compare( index, size, ss.str() ) ) {
    unsigned short index = sistrip::controlView_.size() + sistrip::dir_.size();
    // Extract FEC crate
    if ( !directory.compare( index, sistrip::fecCrate_.size(), sistrip::fecCrate_ ) ) {
      index += sistrip::fecCrate_.size();
      string fec_crate( directory, index, directory.find( directory, index ) - index );
      path.fecCrate_ = atoi( fec_crate.c_str() );
      index = directory.find( sistrip::dir_, index ) + 1;
      // Extract FEC slot
      if ( !directory.compare( index, sistrip::fecSlot_.size(), sistrip::fecSlot_ ) ) {
	index += sistrip::fecSlot_.size();
	string fec_slot( directory, index, directory.find( directory, index ) - index );
	path.fecSlot_ = atoi( fec_slot.c_str() );
	index = directory.find( sistrip::dir_, index ) + 1;
	// Extract FEC ring
	if ( !directory.compare( index, sistrip::fecRing_.size(), sistrip::fecRing_ ) ) {
	  index += sistrip::fecRing_.size();
	  string fec_ring( directory, index, directory.find( directory, index ) - index );
	  path.fecRing_ = atoi( fec_ring.c_str() );
	  index = directory.find( sistrip::dir_, index ) + 1;
	  // Extract CCU address
	  if ( !directory.compare( index, sistrip::ccuAddr_.size(), sistrip::ccuAddr_ ) ) {
	    index += sistrip::ccuAddr_.size();
	    string ccu_addr( directory, index, directory.find( directory, index ) - index );
	    path.ccuAddr_ = atoi( ccu_addr.c_str() );
	    index = directory.find( sistrip::dir_, index ) + 1;
	    // Extract CCU channel
	    if ( !directory.compare( index, sistrip::ccuChan_.size(), sistrip::ccuChan_ ) ) {
	      index += sistrip::ccuChan_.size();
	      string ccu_chan( directory, index, directory.find( directory, index ) - index );
	      path.ccuChan_ = atoi( ccu_chan.c_str() );
	      index = directory.find( sistrip::dir_, index ) + 1;
	    }
	  }
	}
      }
    }
  } else {
    edm::LogError("DQM") << "[SiStripHistoNamingScheme::controlPath]" 
			 << " Unexpected view! Not " << sistrip::controlView_ << "!";
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
  folder << sistrip::readoutView_;
  if ( fed_id != sistrip::all_ ) {
    folder << sistrip::dir_ << sistrip::fedId_ << fed_id;
    if ( fed_channel != sistrip::all_ ) {
      folder << sistrip::dir_ << sistrip::fedChannel_ << fed_channel;
    }
  }
  return folder.str();
}

// -----------------------------------------------------------------------------
// 
string SiStripHistoNamingScheme::histoTitle( Task        histo_task, 
					     Contents    histo_contents,
					     KeyType     key_type,
					     uint32_t    key_value,
					     Granularity granularity,
					     uint16_t    channel,
					     string      extra_info ) {
  
  stringstream title;

  stringstream task;
  if      ( histo_task == SiStripHistoNamingScheme::PEDESTALS )     { task << sistrip::pedestals_; }
  else if ( histo_task == SiStripHistoNamingScheme::OPTO_SCAN )     { task << sistrip::optoScan_; }
  else if ( histo_task == SiStripHistoNamingScheme::APV_TIMING )    { task << sistrip::apvTiming_; }
  else if ( histo_task == SiStripHistoNamingScheme::APV_LATENCY )   { task << sistrip::apvLatency_; }
  else if ( histo_task == SiStripHistoNamingScheme::FED_TIMING )    { task << sistrip::fedTiming_; }
  else if ( histo_task == SiStripHistoNamingScheme::FED_CABLING )   { task << sistrip::fedCabling_; }
  else if ( histo_task == SiStripHistoNamingScheme::VPSP_SCAN )     { task << sistrip::vpspScan_; }
  else if ( histo_task == SiStripHistoNamingScheme::NO_TASK )       { /* add nothing */ }
  else if ( histo_task == SiStripHistoNamingScheme::UNKNOWN_TASK )  { task << sistrip::sep_ << sistrip::unknownTask_; }
  else { edm::LogError("DQM") << "[SiStripHistoNamingScheme::histoTitle]"
			      << " Unexpected histogram task!"; }
  title << task.str();

  stringstream contents;
  if      ( histo_contents == SiStripHistoNamingScheme::SUM2 )             { contents << sistrip::sep_ << sistrip::sum2_; }
  else if ( histo_contents == SiStripHistoNamingScheme::SUM )              { contents << sistrip::sep_ << sistrip::sum_; }
  else if ( histo_contents == SiStripHistoNamingScheme::NUM )              { contents << sistrip::sep_ << sistrip::num_; }
  else if ( histo_contents == SiStripHistoNamingScheme::COMBINED )         { /* add nothing */ }
  else if ( histo_contents == SiStripHistoNamingScheme::UNKNOWN_CONTENTS ) { contents << sistrip::sep_ << sistrip::unknownContents_; }
  else { edm::LogError("DQM") << "[SiStripHistoNamingScheme::histoTitle]"
			      << " Unexpected histogram contents!"; }
  title << contents.str();
  
  stringstream key;
  if      ( key_type == SiStripHistoNamingScheme::FED )         { key << sistrip::sep_ << sistrip::fedKey_ << setfill('0') << setw(8) << hex << key_value; }
  else if ( key_type == SiStripHistoNamingScheme::FEC )         { key << sistrip::sep_ << sistrip::fecKey_ << setfill('0') << setw(8) << hex << key_value; }
  else if ( key_type == SiStripHistoNamingScheme::DET )         { key << sistrip::sep_ << sistrip::detKey_ << setfill('0') << setw(8) << hex << key_value; }
  else if ( key_type == SiStripHistoNamingScheme::NO_KEY )      { /* add nothing */ }
  else if ( key_type == SiStripHistoNamingScheme::UNKNOWN_KEY ) { key << sistrip::sep_ << sistrip::unknownKey_; }
  else { edm::LogError("DQM") << "[SiStripHistoNamingScheme::histoTitle]"
			      << " Unexpected key type!"; }
  title << key.str();

  stringstream gran;
  if      ( granularity == SiStripHistoNamingScheme::LLD_CHAN )     { gran << sistrip::sep_ << sistrip::lldChan_ << channel; }
  else if ( granularity == SiStripHistoNamingScheme::APV_PAIR )     { gran << sistrip::sep_ << sistrip::apvPair_ << channel; }
  else if ( granularity == SiStripHistoNamingScheme::APV )          { gran << sistrip::sep_ << sistrip::apv_     << channel; }
  else if ( granularity == SiStripHistoNamingScheme::MODULE )       { /* add nothing */ }
  else if ( granularity == SiStripHistoNamingScheme::UNKNOWN_GRAN ) { gran << sistrip::sep_ << sistrip::unknownGranularity_; }
  else { edm::LogError("DQM") << "[SiStripHistoNamingScheme::histoTitle]"
			      << " Unexpected granularity!"; }
  title << gran.str();

  if ( extra_info != "" ) { title << sistrip::sep_ << extra_info; }
  
  LogDebug("DQM") << "[SiStripHistoNamingScheme::histoTitle] " << title.str();
  return title.str();
  
}

// -----------------------------------------------------------------------------
// 
SiStripHistoNamingScheme::HistoTitle SiStripHistoNamingScheme::histoTitle( string histo_title ) {
  
  HistoTitle title;
  title.task_        = SiStripHistoNamingScheme::UNKNOWN_TASK;
  title.contents_    = SiStripHistoNamingScheme::UNKNOWN_CONTENTS;
  title.keyType_     = SiStripHistoNamingScheme::UNKNOWN_KEY;
  title.keyValue_    = 0;
  title.granularity_ = SiStripHistoNamingScheme::UNKNOWN_GRAN;
  title.channel_     = 0;
  title.extraInfo_   = "";

  uint32_t position = 0;

  // Extract task 
  if ( histo_title.find( sistrip::fedCabling_, position ) != string::npos ) { 
    title.task_ = SiStripHistoNamingScheme::FED_CABLING; 
    position = histo_title.find( sistrip::fedCabling_, position ) + sistrip::fedCabling_.size();
  } else if ( histo_title.find( sistrip::apvTiming_, position ) != string::npos ) {
    title.task_ = SiStripHistoNamingScheme::APV_TIMING; 
    position = histo_title.find( sistrip::apvTiming_, position ) + sistrip::apvTiming_.size();
  } else if ( histo_title.find( sistrip::fedTiming_, position ) != string::npos ) { 
    title.task_ = SiStripHistoNamingScheme::FED_TIMING;
    position = histo_title.find( sistrip::fedTiming_, position ) + sistrip::fedTiming_.size();
  } else if ( histo_title.find( sistrip::optoScan_, position ) != string::npos ) { 
    title.task_ = SiStripHistoNamingScheme::OPTO_SCAN;
    position = histo_title.find( sistrip::optoScan_, position ) + sistrip::optoScan_.size();
  } else if ( histo_title.find( sistrip::vpspScan_, position ) != string::npos ) { 
    title.task_ = SiStripHistoNamingScheme::VPSP_SCAN;
    position = histo_title.find( sistrip::vpspScan_, position ) + sistrip::vpspScan_.size();
  } else if ( histo_title.find( sistrip::pedestals_, position ) != string::npos ) { 
    title.task_ = SiStripHistoNamingScheme::PEDESTALS;
    position = histo_title.find( sistrip::pedestals_, position ) + sistrip::pedestals_.size();
  } else if ( histo_title.find( sistrip::apvLatency_, position ) != string::npos ) { 
    title.task_ = SiStripHistoNamingScheme::APV_LATENCY;
    position = histo_title.find( sistrip::apvLatency_, position ) + sistrip::apvLatency_.size();
  } else if ( histo_title.find( sistrip::unknownTask_, position ) != string::npos ) { 
    title.task_ = SiStripHistoNamingScheme::UNKNOWN_TASK;
    position = histo_title.find( sistrip::unknownTask_, position ) + sistrip::unknownTask_.size();
  } else { 
    title.task_ = SiStripHistoNamingScheme::NO_TASK; 
  } 

  // Extract contents
  if ( histo_title.find( sistrip::sum2_, position ) != string::npos ) { 
    title.contents_ = SiStripHistoNamingScheme::SUM2;
    position = histo_title.find( sistrip::sum2_, position ) + sistrip::sum2_.size();
  } else if ( histo_title.find( sistrip::sum_, position ) != string::npos ) { 
    title.contents_ = SiStripHistoNamingScheme::SUM;
    position = histo_title.find( sistrip::sum_, position ) + sistrip::sum_.size();
  } else if ( histo_title.find( sistrip::num_, position ) != string::npos ) { 
    title.contents_ = SiStripHistoNamingScheme::NUM;
    position = histo_title.find( sistrip::num_, position ) + sistrip::num_.size();
  } else if ( histo_title.find( sistrip::unknownContents_, position ) != string::npos ) { 
    title.contents_ = SiStripHistoNamingScheme::UNKNOWN_CONTENTS;
    position = histo_title.find( sistrip::unknownContents_, position ) + sistrip::unknownContents_.size();
  } else { 
    title.contents_ = SiStripHistoNamingScheme::COMBINED;
  }
  
  // Extract key type and value
  if ( histo_title.find( sistrip::fedKey_, position ) != string::npos ) { 
    title.keyType_ = SiStripHistoNamingScheme::FED; 
    position = histo_title.find( sistrip::fedKey_, position ) + sistrip::fedKey_.size();
  } else if ( histo_title.find( sistrip::fecKey_, position ) != string::npos ) { 
    title.keyType_ = SiStripHistoNamingScheme::FEC; 
    position = histo_title.find( sistrip::fecKey_, position ) + sistrip::fecKey_.size();
  } else if ( histo_title.find( sistrip::detKey_, position ) != string::npos ) { 
    title.keyType_ = SiStripHistoNamingScheme::DET; 
    position = histo_title.find( sistrip::detKey_, position ) + sistrip::detKey_.size();
  } else if ( histo_title.find( sistrip::unknownKey_, position ) != string::npos ) { 
    title.keyType_ = SiStripHistoNamingScheme::UNKNOWN_KEY; 
    position = histo_title.find( sistrip::unknownKey_, position ) + sistrip::unknownKey_.size();
  } else { 
    title.keyType_ = SiStripHistoNamingScheme::NO_KEY;
  }
  if ( title.keyType_ != SiStripHistoNamingScheme::NO_KEY && 
       title.keyType_ != SiStripHistoNamingScheme::UNKNOWN_KEY ) { 
    stringstream ss; ss << histo_title.substr( position, 8 );
    ss >> hex >> title.keyValue_;
    position += 8;
  } 
  
  // Extract granularity and channel number
  if ( histo_title.find( sistrip::lldChan_, position ) != string::npos ) { 
    title.granularity_ = SiStripHistoNamingScheme::LLD_CHAN; 
    position = histo_title.find( sistrip::lldChan_, position ) + sistrip::lldChan_.size();
  } else if ( histo_title.find( sistrip::apvPair_, position ) != string::npos ) { 
    title.granularity_ = SiStripHistoNamingScheme::APV_PAIR; 
    position = histo_title.find( sistrip::apvPair_, position ) + sistrip::apvPair_.size();
  } else if ( histo_title.find( sistrip::apv_, position ) != string::npos ) { 
    title.granularity_ = SiStripHistoNamingScheme::APV; 
    position = histo_title.find( sistrip::apv_, position ) + sistrip::apv_.size();
  } else if ( histo_title.find( sistrip::unknownGranularity_, position ) != string::npos ) { 
    title.granularity_ = SiStripHistoNamingScheme::UNKNOWN_GRAN; 
    position = histo_title.find( sistrip::unknownGranularity_, position ) + sistrip::unknownGranularity_.size(); 
  } else { 
    title.granularity_ = SiStripHistoNamingScheme::MODULE;
  }
  if ( title.granularity_ != SiStripHistoNamingScheme::MODULE &&
       title.granularity_ != SiStripHistoNamingScheme:: UNKNOWN_GRAN ) { 
    stringstream ss; 
    ss << histo_title.substr( position, histo_title.find( sistrip::sep_, position ) - position );
    ss >> dec >> title.channel_;
    position += ss.str().size();
  } 
  
  // Extract any extra info
  if ( histo_title.find( sistrip::sep_, position ) != string::npos ) { 
    title.extraInfo_ = histo_title.substr( histo_title.find( sistrip::sep_, position )+1, string::npos ); 
  }
  
  // Return HistoTitle struct
  return title;
  
}

// -----------------------------------------------------------------------------
// 
string SiStripHistoNamingScheme::task( SiStripHistoNamingScheme::Task task ) {
  if      ( task == SiStripHistoNamingScheme::PEDESTALS )     { return sistrip::pedestals_; }
  else if ( task == SiStripHistoNamingScheme::OPTO_SCAN )     { return sistrip::optoScan_; }
  else if ( task == SiStripHistoNamingScheme::APV_TIMING )    { return sistrip::apvTiming_; }
  else if ( task == SiStripHistoNamingScheme::APV_LATENCY )   { return sistrip::apvLatency_; }
  else if ( task == SiStripHistoNamingScheme::FED_TIMING )    { return sistrip::fedTiming_; }
  else if ( task == SiStripHistoNamingScheme::FED_CABLING )   { return sistrip::fedCabling_; }
  else if ( task == SiStripHistoNamingScheme::VPSP_SCAN )     { return sistrip::vpspScan_; }
  else if ( task == SiStripHistoNamingScheme::NO_TASK )       { return ""; }
  else if ( task == SiStripHistoNamingScheme::UNKNOWN_TASK )  { return sistrip::unknownTask_; }
  else { 
    edm::LogError("DQM") << "[SiStripHistoNamingScheme::task]"
			 << " Unexpected histogram task!"; 
    return "";
  }
}

// -----------------------------------------------------------------------------
// 
SiStripHistoNamingScheme::Task SiStripHistoNamingScheme::task( string task ) {
  if      ( task == "" )                   { return SiStripHistoNamingScheme::NO_TASK; }
  else if ( task == sistrip::fedCabling_ ) { return SiStripHistoNamingScheme::FED_CABLING; }
  else if ( task == sistrip::apvTiming_ )  { return SiStripHistoNamingScheme::APV_TIMING; }
  else if ( task == sistrip::fedTiming_ )  { return SiStripHistoNamingScheme::FED_TIMING; }
  else if ( task == sistrip::optoScan_ )   { return SiStripHistoNamingScheme::OPTO_SCAN; }
  else if ( task == sistrip::vpspScan_ )   { return SiStripHistoNamingScheme::VPSP_SCAN; }
  else if ( task == sistrip::pedestals_ )  { return SiStripHistoNamingScheme::PEDESTALS; }
  else if ( task == sistrip::apvLatency_ ) { return SiStripHistoNamingScheme::APV_LATENCY; }
  else { return SiStripHistoNamingScheme::UNKNOWN_TASK; }
}  

// -----------------------------------------------------------------------------
// 
string SiStripHistoNamingScheme::contents( SiStripHistoNamingScheme::Contents contents ) {
  if      ( contents == SiStripHistoNamingScheme::COMBINED )          { return ""; }
  else if ( contents == SiStripHistoNamingScheme::SUM2 )              { return sistrip::sum2_; }
  else if ( contents == SiStripHistoNamingScheme::SUM )               { return sistrip::sum_; }
  else if ( contents == SiStripHistoNamingScheme::NUM )               { return sistrip::num_; }
  else if ( contents == SiStripHistoNamingScheme::UNKNOWN_CONTENTS )  { return sistrip::unknownContents_; }
  else { 
    edm::LogError("DQM") << "[SiStripHistoNamingScheme::contents]"
			 << " Unexpected histogram contents!"; 
    return "";
  }
}

// -----------------------------------------------------------------------------
// 
SiStripHistoNamingScheme::Contents SiStripHistoNamingScheme::contents( string contents ) {
  if      ( contents == "" )             { return SiStripHistoNamingScheme::COMBINED; }
  else if ( contents == sistrip::sum2_ ) { return SiStripHistoNamingScheme::SUM2; }
  else if ( contents == sistrip::sum_ )  { return SiStripHistoNamingScheme::SUM; }
  else if ( contents == sistrip::num_ )  { return SiStripHistoNamingScheme::NUM; }
  else { return SiStripHistoNamingScheme::UNKNOWN_CONTENTS; }
}  

// -----------------------------------------------------------------------------
// 
string SiStripHistoNamingScheme::keyType( SiStripHistoNamingScheme::KeyType key_type ) {
  if      ( key_type == SiStripHistoNamingScheme::NO_KEY )       { return ""; }
  else if ( key_type == SiStripHistoNamingScheme::FED )          { return sistrip::fedKey_; }
  else if ( key_type == SiStripHistoNamingScheme::FEC )          { return sistrip::fecKey_; }
  else if ( key_type == SiStripHistoNamingScheme::DET )          { return sistrip::detKey_; }
  else if ( key_type == SiStripHistoNamingScheme::UNKNOWN_KEY )  { return sistrip::unknownKey_; }
  else { 
    edm::LogError("DQM") << "[SiStripHistoNamingScheme::keyType]"
			 << " Unexpected histogram key type!"; 
    return "";
  }
}

// -----------------------------------------------------------------------------
// 
SiStripHistoNamingScheme::KeyType SiStripHistoNamingScheme::keyType( string key_type ) {
  if      ( key_type == "" )               { return SiStripHistoNamingScheme::NO_KEY; }
  else if ( key_type == sistrip::fedKey_ ) { return SiStripHistoNamingScheme::FED; }
  else if ( key_type == sistrip::fecKey_ ) { return SiStripHistoNamingScheme::FEC; }
  else if ( key_type == sistrip::detKey_ ) { return SiStripHistoNamingScheme::DET; }
  else { return SiStripHistoNamingScheme::UNKNOWN_KEY; }
}  

// -----------------------------------------------------------------------------
// 
string SiStripHistoNamingScheme::granularity( SiStripHistoNamingScheme::Granularity granularity ) {
  if      ( granularity == SiStripHistoNamingScheme::MODULE )       { return ""; }
  else if ( granularity == SiStripHistoNamingScheme::LLD_CHAN )     { return sistrip::lldChan_; }
  else if ( granularity == SiStripHistoNamingScheme::APV_PAIR )     { return sistrip::apvPair_; }
  else if ( granularity == SiStripHistoNamingScheme::APV )          { return sistrip::apv_; }
  else if ( granularity == SiStripHistoNamingScheme::UNKNOWN_GRAN ) { return sistrip::unknownGranularity_; }
  else { 
    edm::LogError("DQM") << "[SiStripHistoNamingScheme::granularity]"
			 << " Unexpected histogram granularity!"; 
    return "";
  }
}

// -----------------------------------------------------------------------------
// 
SiStripHistoNamingScheme::Granularity SiStripHistoNamingScheme::granularity( string granularity ) {
  if      ( granularity == "" )                { return SiStripHistoNamingScheme::MODULE; }
  else if ( granularity == sistrip::lldChan_ ) { return SiStripHistoNamingScheme::LLD_CHAN; }
  else if ( granularity == sistrip::apvPair_ ) { return SiStripHistoNamingScheme::APV_PAIR; }
  else if ( granularity == sistrip::apv_ )     { return SiStripHistoNamingScheme::APV; }
  else { return SiStripHistoNamingScheme::UNKNOWN_GRAN; }
}  

