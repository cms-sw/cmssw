#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace std;

// -----------------------------------------------------------------------------
//
sistrip::View SiStripHistoNamingScheme::view( const string& dir ) {
  if      ( dir.find( sistrip::readoutView_ ) != string::npos ) { return sistrip::READOUT; } 
  else if ( dir.find( sistrip::controlView_ ) != string::npos ) { return sistrip::CONTROL; } 
  else if ( dir.find( sistrip::detectorView_ ) != string::npos ) { return sistrip::DETECTOR; } 
  else { return sistrip::UNKNOWN_VIEW; }
}


// -----------------------------------------------------------------------------
//
const string& SiStripHistoNamingScheme::view( const sistrip::View& view ) {
  static string view_str;
  if      ( view == sistrip::READOUT )  { view_str = sistrip::readoutView_; }
  else if ( view == sistrip::CONTROL )  { view_str = sistrip::controlView_; }
  else if ( view == sistrip::DETECTOR ) { view_str = sistrip::detectorView_; }
  else { view_str = sistrip::unknownView_; }
  return view_str;
}

// -----------------------------------------------------------------------------
//
string SiStripHistoNamingScheme::controlPath( uint16_t fec_crate,
					      uint16_t fec_slot,
					      uint16_t fec_ring,
					      uint16_t ccu_addr,
					      uint16_t ccu_chan ) { 
  
  stringstream folder; 
  //folder.reserve(65536); //@@ possible to reserve space???
  folder << sistrip::root_ << sistrip::dir_ << sistrip::controlView_ << sistrip::dir_;
  if ( fec_crate != sistrip::all_ ) {
    folder << sistrip::fecCrate_ << fec_crate << sistrip::dir_;
    if ( fec_slot != sistrip::all_ ) {
      folder << sistrip::fecSlot_ << fec_slot << sistrip::dir_;
      if ( fec_ring != sistrip::all_ ) {
	folder << sistrip::fecRing_ << fec_ring << sistrip::dir_;
	if ( ccu_addr != sistrip::all_ ) {
	  folder << sistrip::ccuAddr_ << ccu_addr << sistrip::dir_;
	  if ( ccu_chan != sistrip::all_ ) {
	    folder << sistrip::ccuChan_ << ccu_chan << sistrip::dir_;
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
const SiStripHistoNamingScheme::ControlPath& SiStripHistoNamingScheme::controlPath( const string& directory ) {

  static ControlPath path;
  path.fecCrate_ = sistrip::all_;
  path.fecSlot_ = sistrip::all_;
  path.fecRing_ = sistrip::all_;
  path.ccuAddr_ = sistrip::all_;
  path.ccuChan_ = sistrip::all_;

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
  folder << sistrip::root_ << sistrip::dir_ << sistrip::readoutView_ << sistrip::dir_;
  if ( fed_id != sistrip::all_ ) {
    folder << sistrip::fedId_ << fed_id << sistrip::dir_;
    if ( fed_channel != sistrip::all_ ) {
      folder << sistrip::fedChannel_ << fed_channel << sistrip::dir_;
    }
  }
  return folder.str();
}

// -----------------------------------------------------------------------------
// 
string SiStripHistoNamingScheme::histoTitle( sistrip::Task        histo_task, 
					     sistrip::Contents    histo_contents,
					     sistrip::KeyType     key_type,
					     uint32_t             key_value,
					     sistrip::Granularity granularity,
					     uint16_t             channel,
					     string               extra_info ) {
  
  stringstream title;

  stringstream task;
  if      ( histo_task == sistrip::PEDESTALS )     { task << sistrip::pedestals_; }
  else if ( histo_task == sistrip::OPTO_SCAN )     { task << sistrip::optoScan_; }
  else if ( histo_task == sistrip::APV_TIMING )    { task << sistrip::apvTiming_; }
  else if ( histo_task == sistrip::APV_LATENCY )   { task << sistrip::apvLatency_; }
  else if ( histo_task == sistrip::FED_TIMING )    { task << sistrip::fedTiming_; }
  else if ( histo_task == sistrip::FED_CABLING )   { task << sistrip::fedCabling_; }
  else if ( histo_task == sistrip::VPSP_SCAN )     { task << sistrip::vpspScan_; }
  else if ( histo_task == sistrip::NO_TASK )       { /* add nothing */ }
  else if ( histo_task == sistrip::UNKNOWN_TASK )  { task << sistrip::sep_ << sistrip::unknownTask_; }
  else { edm::LogError("DQM") << "[SiStripHistoNamingScheme::histoTitle]"
			      << " Unexpected histogram task!"; }
  title << task.str();

  stringstream contents;
  if      ( histo_contents == sistrip::SUM2 )             { contents << sistrip::sep_ << sistrip::sum2_; }
  else if ( histo_contents == sistrip::SUM )              { contents << sistrip::sep_ << sistrip::sum_; }
  else if ( histo_contents == sistrip::NUM )              { contents << sistrip::sep_ << sistrip::num_; }
  else if ( histo_contents == sistrip::COMBINED )         { /* add nothing */ }
  else if ( histo_contents == sistrip::UNKNOWN_CONTENTS ) { contents << sistrip::sep_ << sistrip::unknownContents_; }
  else { edm::LogError("DQM") << "[SiStripHistoNamingScheme::histoTitle]"
			      << " Unexpected histogram contents!"; }
  title << contents.str();
  
  stringstream key;
  if      ( key_type == sistrip::FED )         { key << sistrip::sep_ << sistrip::fedKey_ << setfill('0') << setw(8) << hex << key_value; }
  else if ( key_type == sistrip::FEC )         { key << sistrip::sep_ << sistrip::fecKey_ << setfill('0') << setw(8) << hex << key_value; }
  else if ( key_type == sistrip::DET )         { key << sistrip::sep_ << sistrip::detKey_ << setfill('0') << setw(8) << hex << key_value; }
  else if ( key_type == sistrip::NO_KEY )      { /* add nothing */ }
  else if ( key_type == sistrip::UNKNOWN_KEY ) { key << sistrip::sep_ << sistrip::unknownKey_; }
  else { edm::LogError("DQM") << "[SiStripHistoNamingScheme::histoTitle]"
			      << " Unexpected key type!"; }
  title << key.str();

  stringstream gran;
  if      ( granularity == sistrip::LLD_CHAN )     { gran << sistrip::sep_ << sistrip::lldChan_ << channel; }
  else if ( granularity == sistrip::APV_PAIR )     { gran << sistrip::sep_ << sistrip::apvPair_ << channel; }
  else if ( granularity == sistrip::APV )          { gran << sistrip::sep_ << sistrip::apv_     << channel; }
  else if ( granularity == sistrip::MODULE )       { /* add nothing */ }
  else if ( granularity == sistrip::UNKNOWN_GRAN ) { gran << sistrip::sep_ << sistrip::unknownGranularity_; }
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
  title.task_        = sistrip::UNKNOWN_TASK;
  title.contents_    = sistrip::UNKNOWN_CONTENTS;
  title.keyType_     = sistrip::UNKNOWN_KEY;
  title.keyValue_    = 0;
  title.granularity_ = sistrip::UNKNOWN_GRAN;
  title.channel_     = 0;
  title.extraInfo_   = "";

  uint32_t position = 0;

  // Extract task 
  if ( histo_title.find( sistrip::fedCabling_, position ) != string::npos ) { 
    title.task_ = sistrip::FED_CABLING; 
    position = histo_title.find( sistrip::fedCabling_, position ) + sistrip::fedCabling_.size();
  } else if ( histo_title.find( sistrip::apvTiming_, position ) != string::npos ) {
    title.task_ = sistrip::APV_TIMING; 
    position = histo_title.find( sistrip::apvTiming_, position ) + sistrip::apvTiming_.size();
  } else if ( histo_title.find( sistrip::fedTiming_, position ) != string::npos ) { 
    title.task_ = sistrip::FED_TIMING;
    position = histo_title.find( sistrip::fedTiming_, position ) + sistrip::fedTiming_.size();
  } else if ( histo_title.find( sistrip::optoScan_, position ) != string::npos ) { 
    title.task_ = sistrip::OPTO_SCAN;
    position = histo_title.find( sistrip::optoScan_, position ) + sistrip::optoScan_.size();
  } else if ( histo_title.find( sistrip::vpspScan_, position ) != string::npos ) { 
    title.task_ = sistrip::VPSP_SCAN;
    position = histo_title.find( sistrip::vpspScan_, position ) + sistrip::vpspScan_.size();
  } else if ( histo_title.find( sistrip::pedestals_, position ) != string::npos ) { 
    title.task_ = sistrip::PEDESTALS;
    position = histo_title.find( sistrip::pedestals_, position ) + sistrip::pedestals_.size();
  } else if ( histo_title.find( sistrip::apvLatency_, position ) != string::npos ) { 
    title.task_ = sistrip::APV_LATENCY;
    position = histo_title.find( sistrip::apvLatency_, position ) + sistrip::apvLatency_.size();
  } else if ( histo_title.find( sistrip::unknownTask_, position ) != string::npos ) { 
    title.task_ = sistrip::UNKNOWN_TASK;
    position = histo_title.find( sistrip::unknownTask_, position ) + sistrip::unknownTask_.size();
  } else { 
    title.task_ = sistrip::NO_TASK; 
  } 

  // Extract contents
  if ( histo_title.find( sistrip::sum2_, position ) != string::npos ) { 
    title.contents_ = sistrip::SUM2;
    position = histo_title.find( sistrip::sum2_, position ) + sistrip::sum2_.size();
  } else if ( histo_title.find( sistrip::sum_, position ) != string::npos ) { 
    title.contents_ = sistrip::SUM;
    position = histo_title.find( sistrip::sum_, position ) + sistrip::sum_.size();
  } else if ( histo_title.find( sistrip::num_, position ) != string::npos ) { 
    title.contents_ = sistrip::NUM;
    position = histo_title.find( sistrip::num_, position ) + sistrip::num_.size();
  } else if ( histo_title.find( sistrip::unknownContents_, position ) != string::npos ) { 
    title.contents_ = sistrip::UNKNOWN_CONTENTS;
    position = histo_title.find( sistrip::unknownContents_, position ) + sistrip::unknownContents_.size();
  } else { 
    title.contents_ = sistrip::COMBINED;
  }
  
  // Extract key type and value
  if ( histo_title.find( sistrip::fedKey_, position ) != string::npos ) { 
    title.keyType_ = sistrip::FED; 
    position = histo_title.find( sistrip::fedKey_, position ) + sistrip::fedKey_.size();
  } else if ( histo_title.find( sistrip::fecKey_, position ) != string::npos ) { 
    title.keyType_ = sistrip::FEC; 
    position = histo_title.find( sistrip::fecKey_, position ) + sistrip::fecKey_.size();
  } else if ( histo_title.find( sistrip::detKey_, position ) != string::npos ) { 
    title.keyType_ = sistrip::DET; 
    position = histo_title.find( sistrip::detKey_, position ) + sistrip::detKey_.size();
  } else if ( histo_title.find( sistrip::unknownKey_, position ) != string::npos ) { 
    title.keyType_ = sistrip::UNKNOWN_KEY; 
    position = histo_title.find( sistrip::unknownKey_, position ) + sistrip::unknownKey_.size();
  } else { 
    title.keyType_ = sistrip::NO_KEY;
  }
  if ( title.keyType_ != sistrip::NO_KEY && 
       title.keyType_ != sistrip::UNKNOWN_KEY ) { 
    stringstream ss; ss << histo_title.substr( position, 8 );
    ss >> hex >> title.keyValue_;
    position += 8;
  } 
  
  // Extract granularity and channel number
  if ( histo_title.find( sistrip::lldChan_, position ) != string::npos ) { 
    title.granularity_ = sistrip::LLD_CHAN; 
    position = histo_title.find( sistrip::lldChan_, position ) + sistrip::lldChan_.size();
  } else if ( histo_title.find( sistrip::apvPair_, position ) != string::npos ) { 
    title.granularity_ = sistrip::APV_PAIR; 
    position = histo_title.find( sistrip::apvPair_, position ) + sistrip::apvPair_.size();
  } else if ( histo_title.find( sistrip::apv_, position ) != string::npos ) { 
    title.granularity_ = sistrip::APV; 
    position = histo_title.find( sistrip::apv_, position ) + sistrip::apv_.size();
  } else if ( histo_title.find( sistrip::unknownGranularity_, position ) != string::npos ) { 
    title.granularity_ = sistrip::UNKNOWN_GRAN; 
    position = histo_title.find( sistrip::unknownGranularity_, position ) + sistrip::unknownGranularity_.size(); 
  } else { 
    title.granularity_ = sistrip::MODULE;
  }
  if ( title.granularity_ != sistrip::MODULE &&
       title.granularity_ != sistrip:: UNKNOWN_GRAN ) { 
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
string SiStripHistoNamingScheme::task( sistrip::Task task ) {
  if      ( task == sistrip::PEDESTALS )     { return sistrip::pedestals_; }
  else if ( task == sistrip::OPTO_SCAN )     { return sistrip::optoScan_; }
  else if ( task == sistrip::APV_TIMING )    { return sistrip::apvTiming_; }
  else if ( task == sistrip::APV_LATENCY )   { return sistrip::apvLatency_; }
  else if ( task == sistrip::FED_TIMING )    { return sistrip::fedTiming_; }
  else if ( task == sistrip::FED_CABLING )   { return sistrip::fedCabling_; }
  else if ( task == sistrip::VPSP_SCAN )     { return sistrip::vpspScan_; }
  else if ( task == sistrip::NO_TASK )       { return ""; }
  else if ( task == sistrip::UNKNOWN_TASK )  { return sistrip::unknownTask_; }
  else { 
    edm::LogError("DQM") << "[SiStripHistoNamingScheme::task]"
			 << " Unexpected histogram task!"; 
    return "";
  }
}

// -----------------------------------------------------------------------------
// 
sistrip::Task SiStripHistoNamingScheme::task( string task ) {
  if      ( task == "" )                   { return sistrip::NO_TASK; }
  else if ( task == sistrip::fedCabling_ ) { return sistrip::FED_CABLING; }
  else if ( task == sistrip::apvTiming_ )  { return sistrip::APV_TIMING; }
  else if ( task == sistrip::fedTiming_ )  { return sistrip::FED_TIMING; }
  else if ( task == sistrip::optoScan_ )   { return sistrip::OPTO_SCAN; }
  else if ( task == sistrip::vpspScan_ )   { return sistrip::VPSP_SCAN; }
  else if ( task == sistrip::pedestals_ )  { return sistrip::PEDESTALS; }
  else if ( task == sistrip::apvLatency_ ) { return sistrip::APV_LATENCY; }
  else { return sistrip::UNKNOWN_TASK; }
}  

// -----------------------------------------------------------------------------
// 
string SiStripHistoNamingScheme::contents( sistrip::Contents contents ) {
  if      ( contents == sistrip::COMBINED )          { return ""; }
  else if ( contents == sistrip::SUM2 )              { return sistrip::sum2_; }
  else if ( contents == sistrip::SUM )               { return sistrip::sum_; }
  else if ( contents == sistrip::NUM )               { return sistrip::num_; }
  else if ( contents == sistrip::UNKNOWN_CONTENTS )  { return sistrip::unknownContents_; }
  else { 
    edm::LogError("DQM") << "[SiStripHistoNamingScheme::contents]"
			 << " Unexpected histogram contents!"; 
    return "";
  }
}

// -----------------------------------------------------------------------------
// 
sistrip::Contents SiStripHistoNamingScheme::contents( string contents ) {
  if      ( contents == "" )             { return sistrip::COMBINED; }
  else if ( contents == sistrip::sum2_ ) { return sistrip::SUM2; }
  else if ( contents == sistrip::sum_ )  { return sistrip::SUM; }
  else if ( contents == sistrip::num_ )  { return sistrip::NUM; }
  else { return sistrip::UNKNOWN_CONTENTS; }
}  

// -----------------------------------------------------------------------------
// 
string SiStripHistoNamingScheme::keyType( sistrip::KeyType key_type ) {
  if      ( key_type == sistrip::NO_KEY )       { return ""; }
  else if ( key_type == sistrip::FED )          { return sistrip::fedKey_; }
  else if ( key_type == sistrip::FEC )          { return sistrip::fecKey_; }
  else if ( key_type == sistrip::DET )          { return sistrip::detKey_; }
  else if ( key_type == sistrip::UNKNOWN_KEY )  { return sistrip::unknownKey_; }
  else { 
    edm::LogError("DQM") << "[SiStripHistoNamingScheme::keyType]"
			 << " Unexpected histogram key type!"; 
    return "";
  }
}

// -----------------------------------------------------------------------------
// 
sistrip::KeyType SiStripHistoNamingScheme::keyType( string key_type ) {
  if      ( key_type == "" )               { return sistrip::NO_KEY; }
  else if ( key_type == sistrip::fedKey_ ) { return sistrip::FED; }
  else if ( key_type == sistrip::fecKey_ ) { return sistrip::FEC; }
  else if ( key_type == sistrip::detKey_ ) { return sistrip::DET; }
  else { return sistrip::UNKNOWN_KEY; }
}  

// -----------------------------------------------------------------------------
// 
string SiStripHistoNamingScheme::granularity( sistrip::Granularity granularity ) {
  if      ( granularity == sistrip::MODULE )       { return ""; }
  else if ( granularity == sistrip::LLD_CHAN )     { return sistrip::lldChan_; }
  else if ( granularity == sistrip::APV_PAIR )     { return sistrip::apvPair_; }
  else if ( granularity == sistrip::APV )          { return sistrip::apv_; }
  else if ( granularity == sistrip::UNKNOWN_GRAN ) { return sistrip::unknownGranularity_; }
  else { 
    edm::LogError("DQM") << "[SiStripHistoNamingScheme::granularity]"
			 << " Unexpected histogram granularity!"; 
    return "";
  }
}

// -----------------------------------------------------------------------------
// 
sistrip::Granularity SiStripHistoNamingScheme::granularity( string granularity ) {
  if      ( granularity == "" )                { return sistrip::MODULE; }
  else if ( granularity == sistrip::lldChan_ ) { return sistrip::LLD_CHAN; }
  else if ( granularity == sistrip::apvPair_ ) { return sistrip::APV_PAIR; }
  else if ( granularity == sistrip::apv_ )     { return sistrip::APV; }
  else { return sistrip::UNKNOWN_GRAN; }
}  

