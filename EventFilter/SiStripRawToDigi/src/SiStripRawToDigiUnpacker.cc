#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToDigiUnpacker.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripEventSummary.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "EventFilter/SiStripRawToDigi/interface/TFHeaderDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Utilities/Timing/interface/TimingReport.h"
#include "interface/shared/include/fed_header.h"
#include "interface/shared/include/fed_trailer.h"
#include "Fed9UUtils.hh"
#include "ICException.hh"
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
SiStripRawToDigiUnpacker::SiStripRawToDigiUnpacker( int16_t appended_bytes, 
						    int16_t fed_buffer_dump_freq, 
						    int16_t fed_event_dump_freq, 
						    int16_t trigger_fed_id,
						    bool using_fed_key  ) :
  headerBytes_( appended_bytes ),
  fedBufferDumpFreq_( fed_buffer_dump_freq ),
  fedEventDumpFreq_( fed_event_dump_freq ),
  triggerFedId_( trigger_fed_id ),
  useFedKey_( using_fed_key ),
  fedEvent_(0),
  event_(0)
{
  LogTrace(mlRawToDigi_)
    << "[SiStripRawToDigiUnpacker::"<<__func__<<"]"
    <<" Constructing object...";
  fedEvent_ = new Fed9U::Fed9UEvent();
}

// -----------------------------------------------------------------------------
/** */
SiStripRawToDigiUnpacker::~SiStripRawToDigiUnpacker() {
  LogTrace(mlRawToDigi_)
    << "[SiStripRawToDigiUnpacker::"<<__func__<<"]"
    << " Destructing object...";
  if ( fedEvent_ ) { delete fedEvent_; }
}

// -----------------------------------------------------------------------------
/** */
void SiStripRawToDigiUnpacker::createDigis( const SiStripFedCabling& cabling,
					    const FEDRawDataCollection& buffers,
					    const SiStripEventSummary& summary,
					    RawDigis& scope_mode,
					    RawDigis& virgin_raw,
					    RawDigis& proc_raw,
					    Digis& zero_suppr ) {

  // Check if FEDs found in cabling map and event data
  if ( cabling.feds().empty() ) {
    edm::LogWarning(mlRawToDigi_)
      << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
      << " No FEDs found in cabling map!";
    // Check which FED ids have non-zero size buffers
    std::pair<int,int> fed_range = FEDNumbering::getSiStripFEDIds();
    std::vector<uint16_t> feds;
    for ( uint16_t ifed = static_cast<uint16_t>(fed_range.first);
	  ifed < static_cast<uint16_t>(fed_range.second); ifed++ ) {
      if ( ifed != triggerFedId_ && 
	   buffers.FEDData( static_cast<int>(ifed) ).size() ) {
	feds.push_back(ifed);
      }
    }
    LogTrace(mlRawToDigi_)
      << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
      << " Found " << feds.size() << " FED buffers with non-zero size!";
  }
  
  // Retrieve FED ids from cabling map and iterate through 
  std::vector<uint16_t>::const_iterator ifed = cabling.feds().begin();
  for ( ; ifed != cabling.feds().end(); ifed++ ) {
    
    // Retrieve FED raw data for given FED 
    const FEDRawData& input = buffers.FEDData( static_cast<int>(*ifed) );
    
    // Some debug on FED buffer size
    if ( event_ == 1 && input.data() ) {
      std::stringstream ss;
      ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
	 << " Found FED id " 
	 << std::setw(4) << std::setfill(' ') << *ifed 
	 << " in FEDRawDataCollection"
	 << " with non-zero pointer 0x" 
	 << std::hex
	 << std::setw(4) << std::setfill('0') << *(input.data())
	 << std::dec
	 << " and size (#char) " 
	 << std::setw(5) << std::setfill(' ') << input.size();
      LogTrace(mlRawToDigi_) << ss.str();
    }	
    
    // Dump of FEDRawData to stdout
    if ( fedBufferDumpFreq_ && !(event_%fedBufferDumpFreq_) ) {
      std::stringstream ss;
      dumpRawData( *ifed, input, ss );
      edm::LogVerbatim(mlRawToDigi_) << ss.str();
    }
    
    // Locate start of FED buffer within raw data
    FEDRawData output; 
    locateStartOfFedBuffer( *ifed, input, output );
    
    // Recast data to suit Fed9UEvent
    Fed9U::u32* data_u32 = reinterpret_cast<Fed9U::u32*>( const_cast<unsigned char*>( output.data() ) );
    Fed9U::u32  size_u32 = static_cast<Fed9U::u32>( output.size() / 4 ); 
    
    // Check on FEDRawData pointer
    if ( !data_u32 ) {
      edm::LogWarning(mlRawToDigi_)
	<< "[SiStripRawToDigiUnpacker::" << __func__ << "]"
	<< " NULL pointer to FEDRawData for FED id " << *ifed;
      continue;
    }	
    
    // Check on FEDRawData size
    if ( !size_u32 ) {
      edm::LogWarning(mlRawToDigi_)
	<< "[SiStripRawToDigiUnpacker::" << __func__ << "]"
	<< " FEDRawData has zero size for FED id " << *ifed;
      continue;
    }
    
    // Initialise Fed9UEvent using present FED buffer
    try {
      fedEvent_->Init( data_u32, 0, size_u32 ); 
      //fedEvent_->checkEvent();
    } catch(...) { 
      handleException( __func__, "Problem when creating Fed9UEvent" ); 
      continue;
    } 
   
    // Retrive readout mode
    sistrip::FedReadoutMode mode = sistrip::UNDEFINED_FED_READOUT_MODE;
    try {
      mode = fedReadoutMode( static_cast<unsigned int>( fedEvent_->getSpecialTrackerEventType() ) );
    } catch(...) { handleException( __func__, "Problem extracting readout mode from Fed9UEvent" ); } 
    
    // Dump of FED buffer
    if ( fedEventDumpFreq_ && !(event_%fedEventDumpFreq_) ) {
      std::stringstream ss;
      fedEvent_->dump( ss );
      edm::LogVerbatim(mlRawToDigi_) << ss.str();
    }
    
    // Iterate through FED channels, extract payload and create Digis
    const std::vector<FedChannelConnection>& conns = cabling.connections(*ifed);
    std::vector<FedChannelConnection>::const_iterator iconn = conns.begin();
    for ( ; iconn != conns.end(); iconn++ ) {

      // Check FedId is non-zero, DetId is valid and channel is connected
      if (!iconn->detId() ||
	  (iconn->detId() == sistrip::invalid32_) ||
	  !iconn->fedId() ||
	  !iconn->isConnected()) { continue; }

      uint16_t channel = iconn->fedCh();
      uint16_t iunit = channel / 12;
      uint16_t ichan = channel % 12;
      uint16_t chan  = 12 * iunit + ichan;
      
      try {
	Fed9U::Fed9UAddress addr;
	addr.setFedChannel( static_cast<unsigned char>( channel ) );
	iunit = addr.getFedFeUnit();
	ichan = addr.getFeUnitChannel();
	chan  = 12*( addr.getFedFeUnit() ) + addr.getFeUnitChannel();
      } catch(...) { 
	handleException( __func__, "Problem using Fed9UAddress" ); 
      } 
      
      // Check if FedId is valid
      if ( !iconn->isConnected() ) { continue; }
      
      // Check DetId is valid (if to be used as key)
      if ( !useFedKey_ && ( !iconn->detId() || iconn->detId() == sistrip::invalid32_ ) ) { continue; }

      // Determine whether FED key is inferred from cabling or channel loop
      SiStripFedKey fed_path;

      uint32_t fed_key = 0;

      if ( summary.runType() == sistrip::FED_CABLING ) {
	fed_path = SiStripFedKey( *ifed, 
				  SiStripFedKey::feUnit(chan),
				  SiStripFedKey::feChan(chan) );
      } else { 
	fed_path = SiStripFedKey( iconn->fedId(), 
				  SiStripFedKey::feUnit(iconn->fedCh()),
				  SiStripFedKey::feChan(iconn->fedCh()) );
      }
      fed_key = fed_path.key();
      
      // Determine whether DetId or FED key should be used to index digi containers
      uint32_t key = ( useFedKey_ || mode == sistrip::FED_SCOPE_MODE ) ? fed_key : iconn->detId();
      
      // Determine APV std::pair number (needed only when using DetId)
      uint16_t ipair = ( useFedKey_ || mode == sistrip::FED_SCOPE_MODE ) ? 0 : iconn->apvPairNumber();

      if ( mode == sistrip::FED_SCOPE_MODE ) {

	edm::DetSet<SiStripRawDigi>& sm = scope_mode.find_or_insert(key);
	std::vector<uint16_t> samples; samples.reserve( 1024 ); // theoretical maximum for scope mode length

	try { 
   	  samples = fedEvent_->feUnit( iunit ).channel( ichan ).getSamples();
	} catch(...) { 
	  std::stringstream sss;
	  sss << "Problem accessing SCOPE_MODE data for FedId/FeUnit/FeChan: " 
	      << *ifed << "/" << iunit << "/" << ichan;
	  handleException( __func__, sss.str() ); 
	} 
	if ( !samples.empty() ) { 
	  sm.data.clear(); sm.data.reserve( samples.size() ); sm.data.resize( samples.size() ); 
	  for ( uint16_t i = 0; i < samples.size(); i++ ) {
	    sm.data[i] = SiStripRawDigi( samples[i] ); 
	  }
	}
      } else if ( mode == sistrip::FED_VIRGIN_RAW ) {

	edm::DetSet<SiStripRawDigi>& vr = virgin_raw.find_or_insert(key);
	std::vector<uint16_t> samples; samples.reserve(256); 

	try {
   	  samples = fedEvent_->channel( iunit, ichan ).getSamples();
	} catch(...) { 
	  std::stringstream sss;
	  sss << "Problem accessing VIRGIN_RAW data for FED id/ch: " 
	      << *ifed << "/" << ichan;
	  handleException( __func__, sss.str() ); 
	} 
	//@@ NEED FIX BELOW: vr.data.size() should be 256 * iconn->nApvPairs()
	if ( !samples.empty() ) { 
	  if ( vr.data.size() < static_cast<uint16_t>(256*(ipair+1)) ) { 
	    vr.data.reserve( 256*(ipair+1) ); vr.data.resize( 256*(ipair+1) ); 
	  }
	  uint16_t physical;
	  uint16_t readout; 
	  for ( uint16_t i = 0; i < samples.size(); i++ ) {
	    physical = i%128;
	    readoutOrder( physical, readout ); // convert from physical to readout order
	    (i/128) ? readout=readout*2+1 : readout=readout*2; // multiplexed data
	    vr.data[ipair*256+i] = SiStripRawDigi( samples[readout] ); 
	  }
	}
	
      } else if ( mode == sistrip::FED_PROC_RAW ) {

	edm::DetSet<SiStripRawDigi>& pr = proc_raw.find_or_insert(key);
	std::vector<uint16_t> samples; samples.reserve(256);

	try {
   	  samples = fedEvent_->channel( iunit, ichan ).getSamples();
	} catch(...) { 
	  std::stringstream sss;
	  sss << "Problem accessing PROC_RAW data for FED id/ch: " 
	      << *ifed << "/" << ichan;
	  handleException( __func__, sss.str() ); 
	} 
	//@@ NEED FIX BELOW: pr.data.size() should be 256 * iconn->nApvPairs()
	if ( !samples.empty() ) { 
	  if ( pr.data.size() < static_cast<uint16_t>(256*(ipair+1)) ) { 
	    pr.data.reserve( 256*(ipair+1) ); pr.data.resize( 256*(ipair+1) ); 
	  }
	  int physical;
	  for ( uint16_t i = 0; i < samples.size(); i++ ) {
	    physical = i%128; 
	    (i/128) ? physical=physical*2+1 : physical=physical*2; // multiplexed data
	    pr.data[ipair*256+i] = SiStripRawDigi( samples[physical] ); 
	  } 
	}

      } else if ( mode == sistrip::FED_ZERO_SUPPR ) { 
	
	edm::DetSet<SiStripDigi>& zs = zero_suppr.find_or_insert(key);
	zs.data.reserve(256); // theoretical maximum (768/3, ie, clusters separated by at least 2 strips)
	try{ 
	  Fed9U::Fed9UEventIterator fed_iter = const_cast<Fed9U::Fed9UEventChannel&>(fedEvent_->channel( iunit, ichan )).getIterator();
	  for (Fed9U::Fed9UEventIterator i = fed_iter+7; i.size() > 0; /**/) {
	    unsigned char first_strip = *i++; // first strip of cluster
	    unsigned char width = *i++;       // cluster width in strips 
	    for ( uint16_t istr = 0; istr < width; istr++) {
	      uint16_t strip = ipair*256 + first_strip + istr;
	      zs.data.push_back( SiStripDigi( strip, static_cast<uint16_t>(*i) ) );
	      *i++; // Iterate to next sample
	    }
	  }
	} catch(...) { 
	  std::stringstream sss;
	  sss << "Problem accessing ZERO_SUPPR data for FED id/ch: " 
	      << *ifed << "/" << ichan;
	  handleException( __func__, sss.str() ); 
	} 

      } else if ( mode == sistrip::FED_ZERO_SUPPR_LITE ) { 
	
	edm::DetSet<SiStripDigi>& zs = zero_suppr.find_or_insert(key);
	zs.data.reserve(256); // theoretical maximum (768/3, ie, clusters separated by at least 2 strips)
	try {
	  Fed9U::Fed9UEventIterator fed_iter = const_cast<Fed9U::Fed9UEventChannel&>(fedEvent_->channel( iunit, ichan )).getIterator();
	  for (Fed9U::Fed9UEventIterator i = fed_iter+2; i.size() > 0; /**/) {
	    unsigned char first_strip = *i++; // first strip of cluster
	    unsigned char width = *i++;       // cluster width in strips 
	    for ( uint16_t istr = 0; istr < width; istr++) {
	      uint16_t strip = ipair*256 + first_strip + istr;
	      zs.data.push_back( SiStripDigi( strip, static_cast<uint16_t>(*i) ) );
	      *i++; // Iterate to next sample
	    }
	  }
	} catch(...) { 
	  std::stringstream sss;
	  sss << "Problem accessing ZERO_SUPPR_LITE data for FED id/ch: " 
	      << *ifed << "/" << ichan;
	  handleException( __func__, sss.str() ); 
	} 
	
      } else { // Unknown readout mode! => assume scope mode
	
	std::stringstream ss;
	ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
	   << " Unknown FED readout mode (" << mode
	   << ")! Assuming SCOPE MODE..."; 
  	edm::LogWarning(mlRawToDigi_) << ss.str();

	edm::DetSet<SiStripRawDigi>& sm = scope_mode.find_or_insert(key);
	std::vector<uint16_t> samples; samples.reserve( 1024 ); // theoretical maximum
	try {
   	  samples = fedEvent_->feUnit( iunit ).channel( ichan ).getSamples();
	} catch(...) { 
	  std::stringstream sss;
	  sss << "Problem accessing data (UNKNOWN FED READOUT MODE) for FED id/ch: " 
	      << *ifed << "/" << ichan;
	  handleException( __func__, sss.str() ); 
	} 
	if ( samples.empty() ) { 
 	  edm::LogWarning(mlRawToDigi_)
 	    << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
 	    << " No SM digis found!"; 
	} else {
	  sm.data.clear(); sm.data.reserve( samples.size() ); sm.data.resize( samples.size() ); 
	  for ( uint16_t i = 0; i < samples.size(); i++ ) {
	    sm.data[i] = SiStripRawDigi( samples[i] ); 
	  }
 	  std::stringstream ss;
 	  ss << "Extracted " << samples.size() 
 	     << " SCOPE MODE digis (samples[0] = " << samples[0] 
 	     << ") from FED id/ch " 
 	     << iconn->fedId() << "/" << iconn->fedCh();
 	  //LogTrace(mlRawToDigi_) << ss.str();
	}

      }

    } // channel loop
  } // fed loop
  
  // Incrememt event counter
  event_++;
  
}

// -----------------------------------------------------------------------------
/** */
void SiStripRawToDigiUnpacker::triggerFed( const FEDRawDataCollection& buffers,
					   SiStripEventSummary& summary,
					   const uint32_t& event ) {
  
  // Pointer to data (recast as 32-bit words) and number of 32-bit words
  uint32_t* data_u32 = 0;
  uint32_t  size_u32 = 0;
  
  if ( triggerFedId_ < 0 ) { // Search mode
    
    uint16_t ifed = 0;
    while ( triggerFedId_ < 0 && 
	    ifed < 1 + FEDNumbering::lastFEDId() ) {
      const FEDRawData& trigger_fed = buffers.FEDData( ifed );
      if ( trigger_fed.data() && trigger_fed.size() ) {
	uint8_t*  temp = const_cast<uint8_t*>( trigger_fed.data() );
	data_u32 = reinterpret_cast<uint32_t*>( temp ) + sizeof(fedh_t)/sizeof(uint32_t) + 1;
	size_u32 = trigger_fed.size()/sizeof(uint32_t) - sizeof(fedh_t)/sizeof(uint32_t) - 1;
	fedt_t* fed_trailer = reinterpret_cast<fedt_t*>( temp + trigger_fed.size() - sizeof(fedt_t) );
	if ( fed_trailer->conscheck == 0xDEADFACE ) { 
	  triggerFedId_ = ifed; 
	  std::stringstream ss;
	  ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
	     << " Search mode for 'trigger FED' activated!"
	     << " Found 'trigger FED' info with id " << triggerFedId_;
	  LogTrace(mlRawToDigi_) << ss.str();
	}
      }
      ifed++;
    }
    if ( triggerFedId_ < 0 ) {
      triggerFedId_ = 0;
      std::stringstream ss;
      ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
	 << " Search mode for 'trigger FED' activated!"
	 << " 'Trigger FED' info not found!";
      edm::LogWarning(mlRawToDigi_) << ss.str();
    }
    
  } else if ( triggerFedId_ > 0 ) { // "Trigger FED" id given in .cfg file
    
    const FEDRawData& trigger_fed = buffers.FEDData( triggerFedId_ );
    if ( trigger_fed.data() && trigger_fed.size() ) {
      uint8_t*  temp = const_cast<uint8_t*>( trigger_fed.data() );
      data_u32 = reinterpret_cast<uint32_t*>( temp ) + sizeof(fedh_t)/sizeof(uint32_t) + 1;
      size_u32 = trigger_fed.size()/sizeof(uint32_t) - sizeof(fedh_t)/sizeof(uint32_t) - 1;
      fedt_t* fed_trailer = reinterpret_cast<fedt_t*>( temp + trigger_fed.size() - sizeof(fedt_t) );
      if ( fed_trailer->conscheck != 0xDEADFACE ) { 
	edm::LogWarning(mlRawToDigi_) 
	  << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
	  << " Unexpected stamp found in DAQ trailer (ie, not 0xDEADFACE)!"
	  << " Buffer appears not to contain 'trigger FED' data!";
	triggerFedId_ = 0; 
      }
    }
      
  } else { 
    triggerFedId_ = 0; 
    data_u32 = 0;
    size_u32 = 0;
  }
  
  // Populate summary object with commissioning information
  if ( triggerFedId_ > 0 ) { 

    // Some checks
    if ( !data_u32 ) {
      std::stringstream ss;
      ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
	 << " NULL pointer to 'trigger FED' data";
      edm::LogWarning(mlRawToDigi_) << ss.str();
      return;
    } 
    if ( size_u32 < sizeof(TFHeaderDescription)/sizeof(uint32_t) ) {
      std::stringstream ss;
      ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
	 << " Unexpected 'Trigger FED' data size [32-bit words]: " << size_u32;
      edm::LogWarning(mlRawToDigi_) << ss.str();
      return;
    }
    
    // Write event-specific data to event
    TFHeaderDescription* header = (TFHeaderDescription*) data_u32;
    summary.event( static_cast<uint32_t>( header->getFedEventNumber()) );
    summary.bx( static_cast<uint32_t>( header->getBunchCrossing()) );
    
    // Write commissioning information to event 
    uint32_t hsize = sizeof(TFHeaderDescription)/sizeof(uint32_t);
    uint32_t* head = &data_u32[hsize];
    summary.commissioningInfo( head, event );
    
  }
  
}

//------------------------------------------------------------------------------
/** 
    Removes any data appended prior to FED buffer and reorders 32-bit words if swapped.
    Pattern matches to find DAQ header:
    DAQ header,  4 MSB, BEO1, with value 0x5
    DAQ header,  4 LSB, Hx$$, with value 0x8 (or 0x0)
    DAQ trailer, 4 MSB, EOE,  with value 0xA
*/
void SiStripRawToDigiUnpacker::locateStartOfFedBuffer( const uint16_t& fed_id,
						       const FEDRawData& input,
						       FEDRawData& output ) {
  
  // Check size of input buffer
  if ( input.size() < 24 ) { 
    output.resize( input.size() ); // Return UNadjusted buffer start position and size
    memcpy( output.data(), input.data(), input.size() );
    std::stringstream ss; 
    ss << "[SiStripRawToDigiUnpacker::" << __func__ << "] "
       << "Input FEDRawData with FED id " << fed_id 
       << " has size " << input.size();
    edm::LogWarning(mlRawToDigi_) << ss.str();
    return;
  } 
  
  // Iterator through buffer to find DAQ header 
  bool found = false;
  uint16_t ichar = 0;
  while ( ichar < input.size()-16 && !found ) {
    uint16_t offset = headerBytes_ < 0 ? ichar : headerBytes_; // Negative value means use "search mode" to find DAQ header
    uint32_t* input_u32   = reinterpret_cast<uint32_t*>( const_cast<unsigned char*>( input.data() ) + offset );
    uint32_t* fed_trailer = reinterpret_cast<uint32_t*>( const_cast<unsigned char*>( input.data() ) + input.size() - 8 );

    if ( (input_u32[0]    & 0xF0000000) == 0x50000000 &&
	 (fed_trailer[0]  & 0xF0000000) == 0xA0000000 && 
	 ((fed_trailer[0] & 0x00FFFFFF) * 0x8) == (input.size() - offset) ) {

      // Found DAQ header at byte position 'offset'
      found = true;
      output.resize( input.size()-offset );
      memcpy( output.data(),         // target
	      input.data()+offset,   // source
	      input.size()-offset ); // nbytes
      if ( headerBytes_ < 0 ) {
	std::stringstream ss;
	ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]" 
	   << " Buffer for FED id " << fed_id 
	   << " has been found at byte position " << offset
	   << " with a size of " << input.size()-offset << " bytes."
	   << " Adjust the configurable 'AppendedBytes' to " << offset;
	LogTrace(mlRawToDigi_) << ss.str();
      }

    } else if ( (input_u32[1]    & 0xF0000000) == 0x50000000 &&
		(fed_trailer[1]  & 0xF0000000) == 0xA0000000 &&
		((fed_trailer[1] & 0x00FFFFFF) * 0x8) == (input.size() - offset) ) {

      // Found DAQ header (with MSB and LSB 32-bit words swapped) at byte position 'offset' 
      found = true;
      output.resize( input.size()-offset );
      uint32_t* output_u32 = reinterpret_cast<uint32_t*>( const_cast<unsigned char*>( output.data() ) );
      uint16_t iter = offset; 
      while ( iter < output.size() / sizeof(uint32_t) ) {
	output_u32[iter] = input_u32[iter+1];
	output_u32[iter+1] = input_u32[iter];
	iter+=2;
      }
      if ( headerBytes_ < 0 ) {
	std::stringstream ss;
	ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]" 
	   << " Buffer (with MSB and LSB 32-bit words swapped) for FED id " << fed_id 
	   << " has been found at byte position " << offset
	   << " with a size of " << output.size() << " bytes."
	   << " Adjust the configurable 'AppendedBytes' to " << offset;
	LogTrace(mlRawToDigi_) << ss.str();
      }

    } else { headerBytes_ < 0 ? found = false : found = true; }
    ichar++;
  }      
  
  // Check size of output buffer
  if ( output.size() == 0 ) { 
    
    // Did not find DAQ header after search => return UNadjusted buffer start position and size
    output.resize( input.size() ); 
    memcpy( output.data(), input.data(), input.size() );
    std::stringstream ss;
    if ( headerBytes_ < 0 ) {
      ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
	 << " DAQ header not found within buffer for FED id: " << fed_id;
    } else {
      uint32_t* input_u32 = reinterpret_cast<uint32_t*>( const_cast<unsigned char*>( input.data() ) );
      ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
	 << " DAQ header not found at expected location for FED id: " << fed_id << std::endl
	 << " First 64-bit word of buffer is 0x"
	 << std::hex 
	 << std::setfill('0') << std::setw(8) << input_u32[0] 
	 << std::setfill('0') << std::setw(8) << input_u32[1] 
	 << std::dec << std::endl
	 << " Adjust 'AppendedBytes' configurable to '-1' to activate 'search mode'";
    }
    edm::LogWarning(mlRawToDigi_) << ss.str();

  } else if ( output.size() < 24 ) { // Found DAQ header after search, but too few words
    
    std::stringstream ss; 
    ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
       << " Unexpected buffer size! FEDRawData with FED id " << fed_id 
       << " has size " << output.size();
    edm::LogWarning(mlRawToDigi_) << ss.str();

  } 
  
}

//------------------------------------------------------------------------------
/** 
    Dumps raw data to stdout (NB: payload is byte-swapped,
    headers/trailer are not).
*/
void SiStripRawToDigiUnpacker::dumpRawData( uint16_t fed_id, 
					    const FEDRawData& buffer,
					    std::stringstream& ss ) {
  ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
     << " Dump of buffer for FED id " <<  fed_id << std::endl
     << " Buffer contains " << buffer.size()
     << " bytes (NB: payload is byte-swapped)" << std::endl;
  uint32_t* buffer_u32 = reinterpret_cast<uint32_t*>( const_cast<unsigned char*>( buffer.data() ) );
  unsigned int empty = 0;

  if ( 0 ) { 

    ss << "Byte->   4 5 6 7 0 1 2 3\n";
    for ( uint32_t i = 0; i < buffer.size()/8; i++ ) {
      unsigned int temp0 = buffer_u32[i*2] & 0xFFFFFFFF;
      unsigned int temp1 = buffer_u32[i*2+1] & 0xFFFFFFFF;
      if ( !temp0 && !temp1 ) { empty++; }
      else { 
	if ( empty ) { 
	  ss << "        [ empty  words ]" << std::endl; 
	  empty = 0; 
	}
	ss << std::dec
	   << std::setfill(' ')  << std::setw(6) << i*8 << ": " 
	   << std::hex 
	   << std::setfill('0') << std::setw(8) << temp0 
	   << std::setfill('0') << std::setw(8) << temp1 
	   << std::dec
	   << std::endl;
      }
    }

  } else {
    
    ss << "  Byte |  <---- Byte order ----<  | Byte" << std::endl;
    ss << "  cntr |  7  6  5  4  3  2  1  0  | cntr" << std::endl;
    for ( uint32_t i = 0; i < buffer.size()/8; i++ ) {
      //if ( i>=20 && ((i+4)<(buffer.size()/8)) ) { continue; }
      uint16_t tmp0 = buffer.data()[i*8+0] & 0xFF;
      uint16_t tmp1 = buffer.data()[i*8+1] & 0xFF;
      uint16_t tmp2 = buffer.data()[i*8+2] & 0xFF;
      uint16_t tmp3 = buffer.data()[i*8+3] & 0xFF;
      uint16_t tmp4 = buffer.data()[i*8+4] & 0xFF;
      uint16_t tmp5 = buffer.data()[i*8+5] & 0xFF;
      uint16_t tmp6 = buffer.data()[i*8+6] & 0xFF;
      uint16_t tmp7 = buffer.data()[i*8+7] & 0xFF;
      if ( !tmp0 && !tmp1 && !tmp2 && !tmp3 &&
	   !tmp4 && !tmp5 && !tmp6 && !tmp7 ) { empty++; }
      else { 
	if ( empty ) { 
	  ss << "         [.." 
	     << std::dec << std::setfill('.') << std::setw(4) << empty 
	     << " null words....]" << std::endl; 
	  empty = 0; 
	}
	ss << std::dec
	   << std::setfill(' ')  << std::setw(6) << i*8+7 << " : " 
	   << std::hex 
	   << std::setfill('0') << std::setw(2) << tmp7 << " " 
	   << std::setfill('0') << std::setw(2) << tmp6 << " " 
	   << std::setfill('0') << std::setw(2) << tmp5 << " " 
	   << std::setfill('0') << std::setw(2) << tmp4 << " " 
	   << std::setfill('0') << std::setw(2) << tmp3 << " " 
	   << std::setfill('0') << std::setw(2) << tmp2 << " " 
	   << std::setfill('0') << std::setw(2) << tmp1 << " " 
	   << std::setfill('0') << std::setw(2) << tmp0 
	   << std::dec
	   << " :" << std::setfill(' ')  << std::setw(6) << i*8 
	   << std::endl;
      }
    }

  }
  ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
     << " End of FED buffer";
}

// -----------------------------------------------------------------------------
// 
void SiStripRawToDigiUnpacker::handleException( std::string method_name,
						std::string extra_info ) { // throw (cms::Exception) {
  method_name = "SiStripRawToDigiUnpacker::" + method_name;
  try {
    throw; // rethrow caught exception to be dealt with below
  } 
  catch ( const cms::Exception& e ) { 
    //throw e; // rethrow cms::Exception to be caught by framework
  }
  catch ( const ICUtils::ICException& e ) {
    std::stringstream ss;
    ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
       << " Caught ICUtils::ICException exception!" << std::endl;
    if ( extra_info != "" ) { 
      ss << " Information: " << extra_info << std::endl;
    }
    ss << " Caught ICUtils::ICException in ["
       << method_name << "] with message:" << std::endl 
       << e.what();
    edm::LogWarning(mlRawToDigi_) << ss.str();
    //throw cms::Exception(mlRawToDigi_) << ss.str();
  }
  catch ( const std::exception& e ) {
    std::stringstream ss;
    ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
       << " Caught std::exception!" << std::endl;
    if ( extra_info != "" ) { 
      ss << " Information: " << extra_info << std::endl;
    }
    ss << " Caught std::exception in ["
       << method_name << "] with message:" << std::endl 
       << e.what();
    edm::LogWarning(mlRawToDigi_) << ss.str();
    //throw cms::Exception(mlRawToDigi_) << ss.str();
  }
  catch (...) {
    std::stringstream ss;
    ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
       << " Caught unknown exception!" << std::endl;
    if ( extra_info != "" ) { 
      ss << " Information: " << extra_info << std::endl;
    }
    ss << "Caught unknown exception in ["
       << method_name << "]" << std::endl;
    edm::LogWarning(mlRawToDigi_) << ss.str();
    //throw cms::Exception(mlRawToDigi_) << ss.str();
  }
}

