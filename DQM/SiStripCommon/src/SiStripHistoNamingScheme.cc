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

const string SiStripHistoNamingScheme::fedCabling_ = "FedCabling";
const string SiStripHistoNamingScheme::apvTiming_ = "ApvTiming";
const string SiStripHistoNamingScheme::fedTiming_ = "FedTiming";
const string SiStripHistoNamingScheme::optoScan_ = "OptoScan";
const string SiStripHistoNamingScheme::vpspScan_ = "VpspScan";
const string SiStripHistoNamingScheme::pedestals_ = "Pedestals";
const string SiStripHistoNamingScheme::apvLatency_ = "ApvLatency";
const string SiStripHistoNamingScheme::unknownTask_ = "UnknownTask";

const string SiStripHistoNamingScheme::fedKey_ = "FedKey";
const string SiStripHistoNamingScheme::fecKey_ = "FecKey";
const string SiStripHistoNamingScheme::detKey_ = "DetId"; //@@ necessary?
const string SiStripHistoNamingScheme::unknownKey_ = "UnknownKey";

const string SiStripHistoNamingScheme::sum2_ = "SumOfSquares";
const string SiStripHistoNamingScheme::sum_ = "SumOfContents";
const string SiStripHistoNamingScheme::num_ = "NumOfEntries";
const string SiStripHistoNamingScheme::unknownContents_ = "UnknownContents";

const string SiStripHistoNamingScheme::lldChan_ = "LldChan";
const string SiStripHistoNamingScheme::apvPair_ = "ApvPair";
const string SiStripHistoNamingScheme::apv_ = "Apv";
const string SiStripHistoNamingScheme::unknownGranularity_ = "UnknownGranularity";

const string SiStripHistoNamingScheme::gain_ = "Gain";
const string SiStripHistoNamingScheme::digital_ = "Digital";

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
  LogDebug("DQM") << "[SiStripHistoNamingScheme::controlPath]  " << folder.str();
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
string SiStripHistoNamingScheme::histoTitle( Task        histo_task, 
					     Contents    histo_contents,
					     KeyType     key_type,
					     uint32_t    key_value,
					     Granularity granularity,
					     uint16_t    channel,
					     string      extra_info ) {
  
  stringstream title;

  stringstream task;
  if      ( histo_task == SiStripHistoNamingScheme::PEDESTALS )     { task << pedestals_; }
  else if ( histo_task == SiStripHistoNamingScheme::OPTO_SCAN )     { task << optoScan_; }
  else if ( histo_task == SiStripHistoNamingScheme::APV_TIMING )    { task << apvTiming_; }
  else if ( histo_task == SiStripHistoNamingScheme::APV_LATENCY )   { task << apvLatency_; }
  else if ( histo_task == SiStripHistoNamingScheme::FED_TIMING )    { task << fedTiming_; }
  else if ( histo_task == SiStripHistoNamingScheme::FED_CABLING )   { task << fedCabling_; }
  else if ( histo_task == SiStripHistoNamingScheme::VPSP_SCAN )     { task << vpspScan_; }
  else if ( histo_task == SiStripHistoNamingScheme::NO_TASK )       { /* add nothing */ }
  else if ( histo_task == SiStripHistoNamingScheme::UNKNOWN_TASK )  { task << sep_ << unknownTask_; }
  else { edm::LogError("DQM") << "[SiStripHistoNamingScheme::histoTitle]"
			      << " Unexpected histogram task!"; }
  title << task.str();

  stringstream contents;
  if      ( histo_contents == SiStripHistoNamingScheme::SUM2 )             { contents << sep_ << sum2_; }
  else if ( histo_contents == SiStripHistoNamingScheme::SUM )              { contents << sep_ << sum_; }
  else if ( histo_contents == SiStripHistoNamingScheme::NUM )              { contents << sep_ << num_; }
  else if ( histo_contents == SiStripHistoNamingScheme::COMBINED )         { /* add nothing */ }
  else if ( histo_contents == SiStripHistoNamingScheme::UNKNOWN_CONTENTS ) { contents << sep_ << unknownContents_; }
  else { edm::LogError("DQM") << "[SiStripHistoNamingScheme::histoTitle]"
			      << " Unexpected histogram contents!"; }
  title << contents.str();
  
  stringstream key;
  if      ( key_type == SiStripHistoNamingScheme::FED )         { key << sep_ << fedKey_ << setfill('0') << setw(8) << hex << key_value; }
  else if ( key_type == SiStripHistoNamingScheme::FEC )         { key << sep_ << fecKey_ << setfill('0') << setw(8) << hex << key_value; }
  else if ( key_type == SiStripHistoNamingScheme::DET )         { key << sep_ << detKey_ << setfill('0') << setw(8) << hex << key_value; }
  else if ( key_type == SiStripHistoNamingScheme::NO_KEY )      { /* add nothing */ }
  else if ( key_type == SiStripHistoNamingScheme::UNKNOWN_KEY ) { key << sep_ << unknownKey_; }
  else { edm::LogError("DQM") << "[SiStripHistoNamingScheme::histoTitle]"
			      << " Unexpected key type!"; }
  title << key.str();

  stringstream gran;
  if      ( granularity == SiStripHistoNamingScheme::LLD_CHAN )     { gran << sep_ << lldChan_ << channel; }
  else if ( granularity == SiStripHistoNamingScheme::APV_PAIR )     { gran << sep_ << apvPair_ << channel; }
  else if ( granularity == SiStripHistoNamingScheme::APV )          { gran << sep_ << apv_     << channel; }
  else if ( granularity == SiStripHistoNamingScheme::MODULE )       { /* add nothing */ }
  else if ( granularity == SiStripHistoNamingScheme::UNKNOWN_GRAN ) { gran << sep_ << unknownGranularity_; }
  else { edm::LogError("DQM") << "[SiStripHistoNamingScheme::histoTitle]"
			      << " Unexpected granularity!"; }
  title << gran.str();

  if ( extra_info != "" ) { title << sep_ << extra_info; }
  
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
  if ( histo_title.find( fedCabling_, position ) != string::npos ) { 
    title.task_ = SiStripHistoNamingScheme::FED_CABLING; 
    position = histo_title.find( fedCabling_, position ) + fedCabling_.size();
  } else if ( histo_title.find( apvTiming_, position ) != string::npos ) {
    title.task_ = SiStripHistoNamingScheme::APV_TIMING; 
    position = histo_title.find( apvTiming_, position ) + apvTiming_.size();
  } else if ( histo_title.find( fedTiming_, position ) != string::npos ) { 
    title.task_ = SiStripHistoNamingScheme::FED_TIMING;
    position = histo_title.find( fedTiming_, position ) + fedTiming_.size();
  } else if ( histo_title.find( optoScan_, position ) != string::npos ) { 
    title.task_ = SiStripHistoNamingScheme::OPTO_SCAN;
    position = histo_title.find( optoScan_, position ) + optoScan_.size();
  } else if ( histo_title.find( vpspScan_, position ) != string::npos ) { 
    title.task_ = SiStripHistoNamingScheme::VPSP_SCAN;
    position = histo_title.find( vpspScan_, position ) + vpspScan_.size();
  } else if ( histo_title.find( pedestals_, position ) != string::npos ) { 
    title.task_ = SiStripHistoNamingScheme::PEDESTALS;
    position = histo_title.find( pedestals_, position ) + pedestals_.size();
  } else if ( histo_title.find( apvLatency_, position ) != string::npos ) { 
    title.task_ = SiStripHistoNamingScheme::APV_LATENCY;
    position = histo_title.find( apvLatency_, position ) + apvLatency_.size();
  } else if ( histo_title.find( unknownTask_, position ) != string::npos ) { 
    title.task_ = SiStripHistoNamingScheme::UNKNOWN_TASK;
    position = histo_title.find( unknownTask_, position ) + unknownTask_.size();
  } else { 
    title.task_ = SiStripHistoNamingScheme::NO_TASK; 
  } 

  // Extract contents
  if ( histo_title.find( sum2_, position ) != string::npos ) { 
    title.contents_ = SiStripHistoNamingScheme::SUM2;
    position = histo_title.find( sum2_, position ) + sum2_.size();
  } else if ( histo_title.find( sum_, position ) != string::npos ) { 
    title.contents_ = SiStripHistoNamingScheme::SUM;
    position = histo_title.find( sum_, position ) + sum_.size();
  } else if ( histo_title.find( num_, position ) != string::npos ) { 
    title.contents_ = SiStripHistoNamingScheme::NUM;
    position = histo_title.find( num_, position ) + num_.size();
  } else if ( histo_title.find( unknownContents_, position ) != string::npos ) { 
    title.contents_ = SiStripHistoNamingScheme::UNKNOWN_CONTENTS;
    position = histo_title.find( unknownContents_, position ) + unknownContents_.size();
  } else { 
    title.contents_ = SiStripHistoNamingScheme::COMBINED;
  }
  
  // Extract key type and value
  if ( histo_title.find( fedKey_, position ) != string::npos ) { 
    title.keyType_ = SiStripHistoNamingScheme::FED; 
    position = histo_title.find( fedKey_, position ) + fedKey_.size();
  } else if ( histo_title.find( fecKey_, position ) != string::npos ) { 
    title.keyType_ = SiStripHistoNamingScheme::FEC; 
    position = histo_title.find( fecKey_, position ) + fecKey_.size();
  } else if ( histo_title.find( detKey_, position ) != string::npos ) { 
    title.keyType_ = SiStripHistoNamingScheme::DET; 
    position = histo_title.find( detKey_, position ) + detKey_.size();
  } else if ( histo_title.find( unknownKey_, position ) != string::npos ) { 
    title.keyType_ = SiStripHistoNamingScheme::UNKNOWN_KEY; 
    position = histo_title.find( unknownKey_, position ) + unknownKey_.size();
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
  if ( histo_title.find( lldChan_, position ) != string::npos ) { 
    title.granularity_ = SiStripHistoNamingScheme::LLD_CHAN; 
    position = histo_title.find( lldChan_, position ) + lldChan_.size();
  } else if ( histo_title.find( apvPair_, position ) != string::npos ) { 
    title.granularity_ = SiStripHistoNamingScheme::APV_PAIR; 
    position = histo_title.find( apvPair_, position ) + apvPair_.size();
  } else if ( histo_title.find( apv_, position ) != string::npos ) { 
    title.granularity_ = SiStripHistoNamingScheme::APV; 
    position = histo_title.find( apv_, position ) + apv_.size();
  } else if ( histo_title.find( unknownGranularity_, position ) != string::npos ) { 
    title.granularity_ = SiStripHistoNamingScheme::UNKNOWN_GRAN; 
    position = histo_title.find( unknownGranularity_, position ) + unknownGranularity_.size(); 
  } else { 
    title.granularity_ = SiStripHistoNamingScheme::MODULE;
  }
  if ( title.granularity_ != SiStripHistoNamingScheme::MODULE &&
       title.granularity_ != SiStripHistoNamingScheme:: UNKNOWN_GRAN ) { 
    stringstream ss; 
    ss << histo_title.substr( position, histo_title.find( sep_, position ) - position );
    ss >> dec >> title.channel_;
    position += ss.str().size();
  } 
  
  // Extract any extra info
  if ( histo_title.find( sep_, position ) != string::npos ) { 
    title.extraInfo_ = histo_title.substr( histo_title.find( sep_, position )+1, string::npos ); 
  }
  
  // Return HistoTitle struct
  return title;
  
}

// -----------------------------------------------------------------------------
// 
string SiStripHistoNamingScheme::task( SiStripHistoNamingScheme::Task task ) {
  if      ( task == SiStripHistoNamingScheme::PEDESTALS )     { return pedestals_; }
  else if ( task == SiStripHistoNamingScheme::OPTO_SCAN )     { return optoScan_; }
  else if ( task == SiStripHistoNamingScheme::APV_TIMING )    { return apvTiming_; }
  else if ( task == SiStripHistoNamingScheme::APV_LATENCY )   { return apvLatency_; }
  else if ( task == SiStripHistoNamingScheme::FED_TIMING )    { return fedTiming_; }
  else if ( task == SiStripHistoNamingScheme::FED_CABLING )   { return fedCabling_; }
  else if ( task == SiStripHistoNamingScheme::VPSP_SCAN )     { return vpspScan_; }
  else if ( task == SiStripHistoNamingScheme::NO_TASK )       { return ""; }
  else if ( task == SiStripHistoNamingScheme::UNKNOWN_TASK )  { return unknownTask_; }
  else { 
    edm::LogError("DQM") << "[SiStripHistoNamingScheme::task]"
			 << " Unexpected histogram task!"; 
    return "";
  }
}

// -----------------------------------------------------------------------------
// 
SiStripHistoNamingScheme::Task SiStripHistoNamingScheme::task( string task ) {
  if      ( task == "" )          { return SiStripHistoNamingScheme::NO_TASK; }
  else if ( task == fedCabling_ ) { return SiStripHistoNamingScheme::FED_CABLING; }
  else if ( task == apvTiming_ )  { return SiStripHistoNamingScheme::APV_TIMING; }
  else if ( task == fedTiming_ )  { return SiStripHistoNamingScheme::FED_TIMING; }
  else if ( task == optoScan_ )   { return SiStripHistoNamingScheme::OPTO_SCAN; }
  else if ( task == vpspScan_ )   { return SiStripHistoNamingScheme::VPSP_SCAN; }
  else if ( task == pedestals_ )  { return SiStripHistoNamingScheme::PEDESTALS; }
  else if ( task == apvLatency_ ) { return SiStripHistoNamingScheme::APV_LATENCY; }
  else { return SiStripHistoNamingScheme::UNKNOWN_TASK; }
}  

// -----------------------------------------------------------------------------
// 
string SiStripHistoNamingScheme::contents( SiStripHistoNamingScheme::Contents contents ) {
  if      ( contents == SiStripHistoNamingScheme::COMBINED )          { return ""; }
  else if ( contents == SiStripHistoNamingScheme::SUM2 )              { return sum2_; }
  else if ( contents == SiStripHistoNamingScheme::SUM )               { return sum_; }
  else if ( contents == SiStripHistoNamingScheme::NUM )               { return num_; }
  else if ( contents == SiStripHistoNamingScheme::UNKNOWN_CONTENTS )  { return unknownContents_; }
  else { 
    edm::LogError("DQM") << "[SiStripHistoNamingScheme::contents]"
			 << " Unexpected histogram contents!"; 
    return "";
  }
}

// -----------------------------------------------------------------------------
// 
SiStripHistoNamingScheme::Contents SiStripHistoNamingScheme::contents( string contents ) {
  if      ( contents == "" )    { return SiStripHistoNamingScheme::COMBINED; }
  else if ( contents == sum2_ ) { return SiStripHistoNamingScheme::SUM2; }
  else if ( contents == sum_ )  { return SiStripHistoNamingScheme::SUM; }
  else if ( contents == num_ )  { return SiStripHistoNamingScheme::NUM; }
  else { return SiStripHistoNamingScheme::UNKNOWN_CONTENTS; }
}  

// -----------------------------------------------------------------------------
// 
string SiStripHistoNamingScheme::keyType( SiStripHistoNamingScheme::KeyType key_type ) {
  if      ( key_type == SiStripHistoNamingScheme::NO_KEY )       { return ""; }
  else if ( key_type == SiStripHistoNamingScheme::FED )          { return fedKey_; }
  else if ( key_type == SiStripHistoNamingScheme::FEC )          { return fecKey_; }
  else if ( key_type == SiStripHistoNamingScheme::DET )          { return detKey_; }
  else if ( key_type == SiStripHistoNamingScheme::UNKNOWN_KEY )  { return unknownKey_; }
  else { 
    edm::LogError("DQM") << "[SiStripHistoNamingScheme::keyType]"
			 << " Unexpected histogram key type!"; 
    return "";
  }
}

// -----------------------------------------------------------------------------
// 
SiStripHistoNamingScheme::KeyType SiStripHistoNamingScheme::keyType( string key_type ) {
  if      ( key_type == "" )      { return SiStripHistoNamingScheme::NO_KEY; }
  else if ( key_type == fedKey_ ) { return SiStripHistoNamingScheme::FED; }
  else if ( key_type == fecKey_ ) { return SiStripHistoNamingScheme::FEC; }
  else if ( key_type == detKey_ ) { return SiStripHistoNamingScheme::DET; }
  else { return SiStripHistoNamingScheme::UNKNOWN_KEY; }
}  

// -----------------------------------------------------------------------------
// 
string SiStripHistoNamingScheme::granularity( SiStripHistoNamingScheme::Granularity granularity ) {
  if      ( granularity == SiStripHistoNamingScheme::MODULE )       { return ""; }
  else if ( granularity == SiStripHistoNamingScheme::LLD_CHAN )     { return lldChan_; }
  else if ( granularity == SiStripHistoNamingScheme::APV_PAIR )     { return apvPair_; }
  else if ( granularity == SiStripHistoNamingScheme::APV )          { return apv_; }
  else if ( granularity == SiStripHistoNamingScheme::UNKNOWN_GRAN ) { return unknownGranularity_; }
  else { 
    edm::LogError("DQM") << "[SiStripHistoNamingScheme::granularity]"
			 << " Unexpected histogram granularity!"; 
    return "";
  }
}

// -----------------------------------------------------------------------------
// 
SiStripHistoNamingScheme::Granularity SiStripHistoNamingScheme::granularity( string granularity ) {
  if      ( granularity == "" )       { return SiStripHistoNamingScheme::MODULE; }
  else if ( granularity == lldChan_ ) { return SiStripHistoNamingScheme::LLD_CHAN; }
  else if ( granularity == apvPair_ ) { return SiStripHistoNamingScheme::APV_PAIR; }
  else if ( granularity == apv_ )     { return SiStripHistoNamingScheme::APV; }
  else { return SiStripHistoNamingScheme::UNKNOWN_GRAN; }
}  

