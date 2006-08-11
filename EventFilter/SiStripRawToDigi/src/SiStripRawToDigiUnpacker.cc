#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToDigiUnpacker.h"
//
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Utilities/Timing/interface/TimingReport.h"
//
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigis.h"
#include "DataFormats/SiStripDigi/interface/SiStripEventSummary.h"
#include "DataFormats/SiStripDetId/interface/SiStripReadoutKey.h"
//
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
//
#include "EventFilter/SiStripRawToDigi/interface/TFHeaderDescription.h"
#include "interface/shared/include/fed_header.h"
#include "interface/shared/include/fed_trailer.h"
//
#include "Fed9UUtils.hh"
#include "ICException.hh"
//
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace std;

// -----------------------------------------------------------------------------
/** */
SiStripRawToDigiUnpacker::SiStripRawToDigiUnpacker( int16_t appended_bytes, 
						    int16_t dump_frequency, 
						    int16_t trigger_fed_id,
						    bool    using_fed_key  ) :
  headerBytes_( appended_bytes ),
  dumpFrequency_( dump_frequency ),
  triggerFedId_( trigger_fed_id ),
  useFedKey_( using_fed_key ),
  fedEvent_(0),
  event_(0)
{
  edm::LogVerbatim("RawToDigi") << "[SiStripRawToDigiUnpacker::SiStripRawToDigiUnpacker] Constructing object...";
  fedEvent_ = new Fed9U::Fed9UEvent();
}

// -----------------------------------------------------------------------------
/** */
SiStripRawToDigiUnpacker::~SiStripRawToDigiUnpacker() {
  edm::LogVerbatim("RawToDigi") << "[SiStripRawToDigiUnpacker::~SiStripRawToDigiUnpacker] Destructing object...";
  if ( fedEvent_ ) { delete fedEvent_; }
}

// -----------------------------------------------------------------------------
/** */
void SiStripRawToDigiUnpacker::createDigis( const FedCabling& cabling, 
					    const FedBuffers& buffers, 
					    auto_ptr<SiStripDigis>& digis ) {
  static const string method = "SiStripRawToDigiUnpacker::createDigis";

  // Information for the pseudo-digis object
  vector<sistrip::FedBufferFormat> formats;
  vector<sistrip::FedReadoutMode> modes;
  vector<uint8_t> fe_enable_bits;
  vector<uint16_t> appended_bytes; 
  
  formats.resize(1024);
  modes.resize(1024);
  fe_enable_bits.resize(1024);
  appended_bytes.resize(1024);
  
  // Retrieve FED ids from cabling map and iterate through 
  vector<uint16_t>::const_iterator ifed = cabling->feds().begin();
  for ( ; ifed != cabling->feds().end(); ifed++ ) {
    LogDebug("RawToDigi") << "["<<method<<"] Handling FED id: " << *ifed;
    
    // Retrieve FED raw data for given FED 
    const FEDRawData& input = buffers->FEDData( static_cast<int>(*ifed) );
    
    // Locate start of FED buffer within raw data
    Fed9U::u32* data_u32 = 0;
    Fed9U::u32  size_u32 = 0;
    if ( headerBytes_ != 0 ) {
      FEDRawData output; 
      locateStartOfFedBuffer( *ifed, input, output );
      data_u32 = reinterpret_cast<Fed9U::u32*>( const_cast<unsigned char*>( output.data() ) );
      size_u32 = static_cast<Fed9U::u32>( output.size() / 4 ); 
      appended_bytes[*ifed] = input.size() - output.size();
    } else {
      data_u32 = reinterpret_cast<Fed9U::u32*>( const_cast<unsigned char*>( input.data() ) );
      size_u32 = static_cast<Fed9U::u32>( input.size() / 4 ); 
    }
      
    // Check on FEDRawData pointer
    if ( !data_u32 ) {
      edm::LogError("SiStripRawToDigi") << "["<<method<<"] NULL pointer to FEDRawData!";
      continue;
    }	

    // Check on FEDRawData size
    if ( !size_u32 ) {
      edm::LogError("SiStripRawToDigi") << "["<<method<<"] FEDRawData has zero size!";
      continue;
    }	

    // Initialise Fed9UEvent using present FED buffer and retrive readout mode
    try {
      fedEvent_->Init( data_u32, 0, size_u32 ); 
      fedEvent_->checkEvent();
    } catch(...) { handleException( method, "Problem when creating and checking Fed9UEvent" ); } 
    
    // Information for the pseudo-digis object
    try {
      formats[*ifed] = fedBufferFormat( static_cast<uint16_t>( fedEvent_->getSpecialHeaderFormat() ) ); 
      modes[*ifed] = fedReadoutMode( static_cast<uint16_t>( fedEvent_->getSpecialTrackerEventType() ) );
      fe_enable_bits[*ifed] = fedEvent_->getSpecialFeEnableReg();
    } catch(...) { handleException( method, "Problem when using Fed9UEvent" ); } 
    
//     cout << " FedId: " << *ifed
// 	 << " BufferSize: " << ( 4 * size_u32 )
// 	 << hex
// 	 << " BufferFormat: 0x" << setw(8) << setfill('0') << static_cast<uint16_t>( fedEvent_->getSpecialHeaderFormat() )
// 	 << "/"                 << setw(8) << setfill('0') << static_cast<uint16_t>( fedBufferFormat( fedEvent_->getSpecialHeaderFormat() ) )
// 	 << " ReadoutMode: 0x"  << setw(8) << setfill('0') << static_cast<uint16_t>( fedEvent_->getSpecialTrackerEventType() )
// 	 << "/"                 << setw(8) << setfill('0') << static_cast<uint16_t>( fedReadoutMode( fedEvent_->getSpecialTrackerEventType() ) )
// 	 << " FeEnableBits: 0x" << setw(8) << setfill('0') << static_cast<uint16_t>( fedEvent_->getSpecialFeEnableReg() )
// 	 << dec
// 	 << " AppendedBytes: " << ( input.size() - output.size() )
// 	 << endl;
    
  }
  
  // Create SiStripDigis object
  digis = auto_ptr<SiStripDigis>( new SiStripDigis( buffers, 
						    cabling->feds(), 
						    formats,
						    modes,
						    fe_enable_bits,
						    appended_bytes ) );
  
}

// -----------------------------------------------------------------------------
/** */
void SiStripRawToDigiUnpacker::createDigis( const FedCabling& cabling,
					    const FedBuffers& buffers,
					    std::auto_ptr<RawDigis>& scope_mode,
					    std::auto_ptr<RawDigis>& virgin_raw,
					    std::auto_ptr<RawDigis>& proc_raw,
					    std::auto_ptr<Digis>& zero_suppr ) {
  static const string method = "SiStripRawToDigiUnpacker::createDigis";
  
  // Retrieve FED ids from cabling map and iterate through 
  vector<uint16_t>::const_iterator ifed = cabling->feds().begin();
  for ( ; ifed != cabling->feds().end(); ifed++ ) {
    LogDebug("RawToDigi") << "["<<method<<"] Extracting payload from FED id: " << *ifed;
    
    // Retrieve FED raw data for given FED 
    const FEDRawData& input = buffers->FEDData( static_cast<int>(*ifed) );

    // Dump of FEDRawData to stdout
    if ( dumpFrequency_ && !(event_%dumpFrequency_) ) {
      stringstream ss;
      dumpRawData( *ifed, input, ss );
      LogTrace("RawToDigi") << ss.str();
    }
    
    // Locate start of FED buffer within raw data
    FEDRawData output; 
    locateStartOfFedBuffer( *ifed, input, output );
    
    // Recast data to suit Fed9UEvent
    Fed9U::u32* data_u32 = reinterpret_cast<Fed9U::u32*>( const_cast<unsigned char*>( output.data() ) );
    Fed9U::u32  size_u32 = static_cast<Fed9U::u32>( output.size() / 4 ); 
    
    // Check on FEDRawData pointer
    if ( !data_u32 ) {
      edm::LogError("SiStripRawToDigi") << "["<<method<<"] NULL pointer to FEDRawData!";
      continue;
    }	

    // Check on FEDRawData size
    if ( !size_u32 ) {
      edm::LogError("SiStripRawToDigi") << "["<<method<<"] FEDRawData has zero size!";
      continue;
    }	

    // Initialise Fed9UEvent using present FED buffer
    sistrip::FedReadoutMode mode = sistrip::UNDEFINED_FED_READOUT_MODE;
    try {
      fedEvent_->Init( data_u32, 0, size_u32 ); 
      fedEvent_->checkEvent();
    } catch(...) { handleException( method, "Problem unpacking FED buffer" ); } 

    // Retrive readout mode
    try {
      mode = fedReadoutMode( static_cast<unsigned int>( fedEvent_->getSpecialTrackerEventType() ) );
    } catch(...) { handleException( method, "Problem when using Fed9UEvent" ); } 
    
    // Dump of FED buffer
    if ( dumpFrequency_ && !(event_%dumpFrequency_) ) {
      stringstream ss;
      fedEvent_->dump( ss );
      LogTrace("RawToDigi") << ss.str();
    }

    // Iterate through FED channels, extract payload and create Digis
    Fed9U::Fed9UAddress addr;
    for ( uint16_t channel = 0; channel < 96; channel++ ) {
      
      uint16_t iunit = 0;
      uint16_t ichan = 0;
      uint16_t chan = 0;
      try {
	addr.setFedChannel( static_cast<unsigned char>( channel ) );
	iunit = addr.getFedFeUnit();
	ichan = addr.getFeUnitChannel();
	chan = 12*( addr.getFedFeUnit() ) + addr.getFeUnitChannel();
      } catch(...) { 
	handleException( method, "Problem unpacking FED payload" ); 
      } 
      
      // Retrieve cabling map information and define "FED key" for Digis
      const FedChannelConnection& conn = cabling->connection( *ifed, chan );
      
      // Determine whether DetId or FED key should be used to index digi containers
      uint32_t fed_key = SiStripReadoutKey::key( conn.fedId(), conn.fedCh() );
      uint32_t key     = ( useFedKey_ || mode == sistrip::SCOPE_MODE ) ? fed_key : conn.detId();
      uint16_t ipair   = ( useFedKey_ || mode == sistrip::SCOPE_MODE ) ? 0 : conn.apvPairNumber();
      
      // Check for non-zero key OR scope mode
      if ( !key ) { continue; }
      
      if ( mode == sistrip::SCOPE_MODE ) {

	edm::DetSet<SiStripRawDigi>& sm = scope_mode->find_or_insert( key );
	vector<uint16_t> samples; samples.reserve( 1024 ); // theoretical maximum for scope mode length
	samples = fedEvent_->feUnit( iunit ).channel( ichan ).getSamples();
	if ( !samples.empty() ) { 
	  sm.data.clear(); sm.data.reserve( samples.size() ); sm.data.resize( samples.size() ); 
	  for ( uint16_t i = 0; i < samples.size(); i++ ) {
	    sm.data[i] = SiStripRawDigi( samples[i] ); 
	  }
	}
	
      } else if ( mode == sistrip::VIRGIN_RAW ) {

	edm::DetSet<SiStripRawDigi>& vr = virgin_raw->find_or_insert( key );
	vector<uint16_t> samples; samples.reserve(256);
	samples = fedEvent_->channel( iunit, ichan ).getSamples();
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

      } else if ( mode == sistrip::PROC_RAW ) {

	edm::DetSet<SiStripRawDigi>& pr = proc_raw->find_or_insert( key ) ;
	vector<uint16_t> samples; samples.reserve(256);
	samples = fedEvent_->channel( iunit, ichan ).getSamples();
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

      } else if ( mode == sistrip::ZERO_SUPPR ) { 
	
	edm::DetSet<SiStripDigi>& zs = zero_suppr->find_or_insert( key );
	zs.data.reserve(256); // theoretical maximum (768/3, ie, clusters separated by at least 2 strips)
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

      } else if ( mode == sistrip::ZERO_SUPPR_LITE ) { 
	
	edm::DetSet<SiStripDigi>& zs = zero_suppr->find_or_insert( key );
	zs.data.reserve(256); // theoretical maximum (768/3, ie, clusters separated by at least 2 strips)
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
	
      } else { // Unknown readout mode! (=> assume scope mode)
	
	stringstream ss;
	ss << "["<<method<<"]"
	   << " Unknown FED readout mode (" << mode
	   << ")! Assuming SCOPE MODE..."; 
	edm::LogError("RawToDigi") << ss.str();
	edm::DetSet<SiStripRawDigi>& sm = scope_mode->find_or_insert( key );
	vector<uint16_t> samples; samples.reserve( 1024 ); // theoretical maximum
	samples = fedEvent_->feUnit( iunit ).channel( ichan ).getSamples();
	if ( samples.empty() ) { 
	  edm::LogWarning("Commissioning") << "["<<method<<"] No SM digis found!"; 
	} else {
	  sm.data.clear(); sm.data.reserve( samples.size() ); sm.data.resize( samples.size() ); 
	  for ( uint16_t i = 0; i < samples.size(); i++ ) {
	    sm.data[i] = SiStripRawDigi( samples[i] ); 
	  }
	  LogDebug("Commissioning") << "Extracted " << samples.size() 
				    << " SCOPE MODE digis (samples[0] = " << samples[0] 
				    << ") from FED id/ch " 
				    << conn.fedId() << "/" << conn.fedCh();
	}

      }

    } // channel loop
  } // fed loop
  
  // Incrememt event counter
  event_++;
  
}

// -----------------------------------------------------------------------------
/** */
void SiStripRawToDigiUnpacker::triggerFed( const FedBuffers& buffers,
					   auto_ptr<SiStripEventSummary>& summary ) {
  static const string method = "SiStripRawToDigiUnpacker::triggerFed"; 

  // Pointer to data (recast as 32-bit words) and number of 32-bit words
  uint32_t* data_u32 = 0;
  uint32_t  size_u32 = 0;
  
  if ( triggerFedId_ < 0 ) { // Search mode
    
    uint16_t ifed = 0;
    while ( triggerFedId_ < 0 && 
	    ifed < 1 + FEDNumbering::lastFEDId() ) {
      const FEDRawData& trigger_fed = buffers->FEDData( ifed );
      if ( trigger_fed.data() && trigger_fed.size() ) {
	uint8_t*  temp = const_cast<uint8_t*>( trigger_fed.data() );
	data_u32 = reinterpret_cast<uint32_t*>( temp ) + sizeof(fedh_t)/sizeof(uint32_t) + 1;
	size_u32 = trigger_fed.size()/sizeof(uint32_t) - sizeof(fedh_t)/sizeof(uint32_t) - 1;
	fedt_t* fed_trailer = reinterpret_cast<fedt_t*>( temp + trigger_fed.size() - sizeof(fedt_t) );
	if ( fed_trailer->conscheck == 0xDEADFACE ) { 
	  triggerFedId_ = ifed; 
	  stringstream ss;
	  ss << "["<<method<<"]"
	     << " Search mode for 'trigger FED' activated!"
	     << " Found 'trigger FED' info with id " << triggerFedId_;
	  edm::LogVerbatim("RawToDigi") << ss.str();
	}
      }
      ifed++;
    }
    if ( triggerFedId_ < 0 ) {
      triggerFedId_ = 0;
      stringstream ss;
      ss << "["<<method<<"]"
	 << " Search mode for 'trigger FED' activated!"
	 << " 'Trigger FED' info not found!";
      edm::LogWarning("RawToDigi") << ss.str();
    }
    
  } else if ( triggerFedId_ > 0 ) { // "Trigger FED" id given in .cfg file
    
    const FEDRawData& trigger_fed = buffers->FEDData( triggerFedId_ );
      if ( trigger_fed.data() && trigger_fed.size() ) {
	uint8_t*  temp = const_cast<uint8_t*>( trigger_fed.data() );
	data_u32 = reinterpret_cast<uint32_t*>( temp ) + sizeof(fedh_t)/sizeof(uint32_t) + 1;
	size_u32 = trigger_fed.size()/sizeof(uint32_t) - sizeof(fedh_t)/sizeof(uint32_t) - 1;
	fedt_t* fed_trailer = reinterpret_cast<fedt_t*>( temp + trigger_fed.size() - sizeof(fedt_t) );
	if ( fed_trailer->conscheck != 0xDEADFACE ) { triggerFedId_ = 0; }
      }
      
  } else { 
    triggerFedId_ = 0; 
    data_u32 = 0;
    size_u32 = 0;
  }
  
  // Populate summary object with commissioning information
  if ( triggerFedId_ > 0 &&
       data_u32 &&
       size_u32 > sizeof(TFHeaderDescription)/sizeof(uint32_t) ) {
    
    // Write event-specific data to event
    TFHeaderDescription* header = (TFHeaderDescription*) data_u32;
    summary->event( static_cast<uint32_t>( header->getFedEventNumber()) );
    summary->bx( static_cast<uint32_t>( header->getBunchCrossing()) );
      
    // Write commissioning information to event 
    uint32_t hsize = sizeof(TFHeaderDescription)/sizeof(uint32_t);
    uint32_t* head = &data_u32[hsize];
    summary->commissioningInfo( head );

//     stringstream ss;
//     ss << "["<<method<<"]"
//        << "  getBunchCrossing: " << header->getBunchCrossing()
//        << "  getNumberOfChannels: " << header->getNumberOfChannels() 
//        << "  getNumberOfSamples: " << header->getNumberOfSamples()
//        << "  getFedType: 0x" 
//        << hex << setw(8) << setfill('0') << header->getFedType() << dec
//        << "  getFedId: " << header->getFedId()
//        << "  getFedEventNumber: " << header->getFedEventNumber();
//     LogDebug("RawToDigi") << ss.str();
      
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
  static const string method = "SiStripRawToDigiUnpacker::locateStartOfFedBuffer";
  
  // Check size of input buffer
  if ( input.size() < 24 ) { 
    output.resize( input.size() ); // Return UNadjusted buffer start position and size
    memcpy( output.data(), input.data(), input.size() );
    stringstream ss; 
    ss << "["<<method<<"] "
       << "Input FEDRawData with FED id " << fed_id 
       << " has size " << input.size() << "\n";
    edm::LogError("SiStripRawToDigiUnpacker") << ss.str();
    throw cms::Exception("SiStripRawToDigiUnpacker") << ss.str();
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
	stringstream ss;
	ss << "["<<method<<"]" 
	   << " Buffer for FED id " << fed_id 
	   << " has been found at byte position " << offset
	   << " with a size of " << input.size()-offset << " bytes."
	   << " Adjust the configurable 'AppendedBytes' to " << offset;
	edm::LogVerbatim("RawToDigi") << ss.str();
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
	stringstream ss;
	ss << "["<<method<<"]" 
	   << " Buffer (with MSB and LSB 32-bit words swapped) for FED id " << fed_id 
	   << " has been found at byte position " << offset
	   << " with a size of " << output.size() << " bytes."
	   << " Adjust the configurable 'AppendedBytes' to " << offset;
	edm::LogVerbatim("RawToDigi") << ss.str();
      }

    } else { headerBytes_ < 0 ? found = false : found = true; }
    ichar++;
  }      
  
  // Check size of output buffer
  if ( output.size() == 0 ) { 
    
    // Did not find DAQ header after search => return UNadjusted buffer start position and size
    output.resize( input.size() ); 
    memcpy( output.data(), input.data(), input.size() );
    stringstream ss;
    if ( headerBytes_ < 0 ) {
      ss << "["<<method<<"]"
	 << " DAQ header not found within buffer for FED id " << fed_id << "!";
    } else {
      uint32_t* input_u32 = reinterpret_cast<uint32_t*>( const_cast<unsigned char*>( input.data() ) );
      ss << "["<<method<<"]"
	 << " DAQ header not found at expected location for FED id " << fed_id << "!"
	 << " First 64-bit word of buffer is 0x"
	 << hex 
	 << setfill('0') << setw(8) << input_u32[0] 
	 << setfill('0') << setw(8) << input_u32[1] 
	 << dec
	 << ". Adjust 'AppendedBytes' configurable"
	 << " to negative value to activate 'search mode'";
    }
    edm::LogError("SiStripRawToDigiUnpacker") << ss.str();
    //throw cms::Exception("SiStripRawToDigiUnpacker") << ss.str();

  } else if ( output.size() < 24 ) { // Found DAQ header after search, but too few words

    stringstream ss; 
    ss << "["<<method<<"]"
       << " Unexpected buffer size! FEDRawData with FED id " << fed_id 
       << " has size " << output.size();
    edm::LogError("RawToDigi") << ss.str();
    //throw cms::Exception("SiStripRawToDigiUnpacker") << ss.str();

  } 
  
}

//------------------------------------------------------------------------------
/** 
    Dumps raw data to stdout (NB: payload is byte-swapped,
    headers/trailer are not).
*/
void SiStripRawToDigiUnpacker::dumpRawData( uint16_t fed_id, 
					    const FEDRawData& buffer,
					    stringstream& ss ) {
  static const string method = "SiStripRawToDigiUnpacker::dumpRawData";
  ss << "["<<method<<"] Dump of buffer for FED id " <<  fed_id
     << ". Buffer contains " << buffer.size() << " bytes (NB: payload is byte-swapped). \n";
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
	  ss << "        [ empty  words ]\n"; 
	  empty = 0; 
	}
	ss << dec
	   << setfill(' ')  << setw(6) << i*8 << ": " 
	   << hex 
	   << setfill('0') << setw(8) << temp0 
	   << setfill('0') << setw(8) << temp1 
	   << dec
	   << "\n";
      }
    }

  } else {
    
    ss << "  Byte |  <---- byte order ----<  | Byte\n";
    ss << "  cntr |  7  6  5  4  3  2  1  0  | cntr\n";
    for ( uint32_t i = 0; i < buffer.size()/8; i++ ) {

      if ( i>=20 && ((i+4)<(buffer.size()/8)) ) { continue; }

      unsigned int tmp0 = buffer.data()[i*8+0] & 0xFF;
      unsigned int tmp1 = buffer.data()[i*8+1] & 0xFF;
      unsigned int tmp2 = buffer.data()[i*8+2] & 0xFF;
      unsigned int tmp3 = buffer.data()[i*8+3] & 0xFF;
      unsigned int tmp4 = buffer.data()[i*8+4] & 0xFF;
      unsigned int tmp5 = buffer.data()[i*8+5] & 0xFF;
      unsigned int tmp6 = buffer.data()[i*8+6] & 0xFF;
      unsigned int tmp7 = buffer.data()[i*8+7] & 0xFF;
      if ( !tmp0 && !tmp1 && !tmp2 && !tmp3&&
	   !tmp4 && !tmp5 && !tmp6 && !tmp7 ) { empty++; }
      else { 
	if ( empty ) { 
	  ss << "         ......empty words......\n"; 
	  empty = 0; 
	}
	ss << dec
	   << setfill(' ')  << setw(6) << i*8+7 << " : " 
	   << hex 
	   << setfill('0') << setw(2) << tmp7 << " " 
	   << setfill('0') << setw(2) << tmp6 << " " 
	   << setfill('0') << setw(2) << tmp5 << " " 
	   << setfill('0') << setw(2) << tmp4 << " " 
	   << setfill('0') << setw(2) << tmp3 << " " 
	   << setfill('0') << setw(2) << tmp2 << " " 
	   << setfill('0') << setw(2) << tmp1 << " " 
	   << setfill('0') << setw(2) << tmp0 
	   << dec
	   << " :" << setfill(' ')  << setw(6) << i*8 
	   << "\n";
      }
    }

  }
  ss << "["<<method<<"] End of FED buffer";
}

// -----------------------------------------------------------------------------
// 
void SiStripRawToDigiUnpacker::handleException( const string& method_name,
						const string& extra_info ) throw (cms::Exception) {
  try {
    throw; // rethrow caught exception to be dealt with below
  } 
  catch ( const cms::Exception& e ) { 
    throw e; // rethrow cms::Exception to be caught by framework
  }
  catch ( const ICUtils::ICException& e ) {
    stringstream ss;
    ss << "Caught ICUtils::ICException in ["
       << method_name << "] with message: \n" 
       << e.what();
    if ( extra_info != "" ) { ss << "Additional info: " << extra_info; }
    edm::LogError("SiStripRawToDigiUnpacker") << ss.str();
    throw cms::Exception("SiStripRawToDigiUnpacker") << ss.str();
  }
  catch ( const exception& e ) {
    stringstream ss;
    ss << "Caught std::exception in ["
       << method_name << "] with message: \n" 
       << e.what();
    if ( extra_info != "" ) { ss << "Additional info: " << extra_info; }
    edm::LogError("SiStripRawToDigiUnpacker") << ss.str();
    throw cms::Exception("SiStripRawToDigiUnpacker") << ss.str();
  }
  catch (...) {
    stringstream ss;
    ss << "Caught unknown exception in ["
       << method_name << "]";
    if ( extra_info != "" ) { ss << "\n" << "Additional info: " << extra_info; }
    edm::LogError("SiStripRawToDigiUnpacker") << ss.str();
    throw cms::Exception("SiStripRawToDigiUnpacker") << ss.str();
  }
}





//   if ( headerBytes_ < 0 ) { // Use "search mode" to find DAQ header
    
//     for ( uint16_t ichar = 0; ichar < input.size()-16; ichar++ ) { 
//       uint32_t* input_u32 = reinterpret_cast<uint32_t*>( const_cast<unsigned char*>( input.data() ) + ichar );
//       uint32_t* trailer   = reinterpret_cast<uint32_t*>( const_cast<unsigned char*>( input.data() ) + input.size() - 4 );
//       if ( (input_u32[0] & 0xF0000000) == 0x50000000 &&
// 	   (input_u32[1] & 0x0000000F) == 0x00000008 &&
// 	   //(input_u32[2] & 0xFF000000) == 0xED000000 &&
// 	   (*trailer     & 0x0000000F) == 0x0000000a ) {
// 	edm::LogVerbatim("RawToDigi") << "[SiStripRawToDigiUnpacker::locateStartOfFedBuffer]" 
// 				  << " FED buffer has been found at byte position " 
// 				  << ichar << " with a size of " << input.size()-ichar << " bytes";
// 	edm::LogVerbatim("RawToDigi") << "[SiStripRawToDigiUnpacker::locateStartOfFedBuffer]" 
// 				  << " Adjust the configurable 'AppendedHeaderBytes' to " << ichar;
// 	// Found DAQ header at byte position 'ichar' 
// 	// Return adjusted buffer start position and size
// 	output.resize( input.size()-ichar );
// 	memcpy( output.data(),        // target
// 		input.data()+ichar,   // source
// 		input.size()-ichar ); // nbytes
//       }
//     }
//     // Didn't find DAQ header after search
//     // Return UNadjusted buffer start position and size
//     output.resize( input.size() );
//     memcpy( output.data(), input.data(), input.size() );
    
//   } else { 

//     // Adjust according to the 'AppendedHeaderBytes' configurable
//     uint32_t* input_u32 = reinterpret_cast<uint32_t*>( const_cast<unsigned char*>( input.data() ) + headerBytes_ );
//     BOE1 = input_u32[0] & 0xF0000000;
//     HxSS = input_u32[1] & 0x0000000F;
//     Resv = input_u32[2] & 0xFF000000;
//     if ( !( BOE1 == 0x50000000 &&
// 	    HxSS == 0x00000008 ) ) { 
//       stringstream ss;
//       ss << "[SiStripRawToDigiUnpacker::locateStartOfFedBuffer]"
// 	 << " DAQ header not found at expected location!"
// 	 << " First 64-bit word of buffer is 0x"
// 	 << hex 
// 	 << setfill('0') << setw(8) << input_u32[0] 
// 	 << setfill('0') << setw(8) << input_u32[1] 
// 	 << dec
// 	 << ". Adjust 'AppendedHeaderBytes' configurable"
// 	 << " to negative value to activate 'search mode'";
//       edm::LogError("RawToDigi") << ss.str();
//       // DAQ header not found at expected location
//       // Return UNadjusted buffer start position and size
//       output.resize( output.size() );
//       memcpy( output.data(), input.data(), input.size() );
//     } else {
//       // DAQ header found at expected location
//       // Return adjusted buffer start position and size
//       output.resize( input.size()-headerBytes_ );
//       memcpy( output.data(), 
// 	      input.data()+headerBytes_, 
// 	      input.size()-headerBytes_ );
//     }
//   }
  
//   // Check on size of output buffer
//   if ( output.size() < 24 ) { 
//     stringstream ss; 
//     ss << "[SiStripRawToDigiUnpacker::locateStartOfFedBuffer]"
//        << " Unexpected buffer size! FEDRawData with FED id " << fed_id 
//        << " has size " << output.size();
//     edm::LogError("RawToDigi") << ss.str();
//   } 
 









// // -----------------------------------------------------------------------------
// /** */
// void SiStripRawToDigiUnpacker::digiInfo( vector<uint32_t>& keys, //@@ TEMP!
// 				 auto_ptr< edm::DetSetVector<SiStripRawDigi> >& scope_mode,
// 				 auto_ptr< edm::DetSetVector<SiStripRawDigi> >& virgin_raw,
// 				 auto_ptr< edm::DetSetVector<SiStripRawDigi> >& proc_raw,
// 				 auto_ptr< edm::DetSetVector<SiStripDigi> >& zero_suppr ) {
//   LogDebug("RawToDigi") << "[SiStripRawToDigiUnpacker::digiInfo] Number of keys: " << keys.size();
//   vector<uint32_t>::iterator ikey;
//   for ( ikey = keys.begin(); ikey != keys.end(); ikey++ ) {
//     LogDebug("RawToDigi") << "[SiStripRawToDigiUnpacker::digiInfo] Key: " << hex << setfill('0') << setw(8) << *ikey << dec;
//     // SM
//     vector< edm::DetSet<SiStripRawDigi> >::const_iterator sm = scope_mode->find( *ikey );
//     if ( sm != scope_mode->end() ) {
//       if ( sm->data.empty() ) { LogDebug("RawToDigi") << "[SiStripRawToDigiUnpacker::digiInfo] No SM digis found!"; }
//       LogDebug("RawToDigi") << "[SiStripRawToDigiUnpacker::digiInfo] sm->data.size(): " << sm->data.size();
//       for ( uint16_t ism = 0; ism < sm->data.size(); ism++ ) { anal_.smDigi( ism, sm->data[ism].adc() ); }
//     } else { LogDebug("RawToDigi") << "[SiStripRawToDigiUnpacker::digiInfo] Key not found for SM digis"; }
//     // VR
//     vector< edm::DetSet<SiStripRawDigi> >::const_iterator vr = virgin_raw->find( *ikey );
//     if ( vr != virgin_raw->end() ) {
//       if ( vr->data.empty() ) { LogDebug("RawToDigi") << "[SiStripRawToDigiUnpacker::digiInfo] No VR digis found!"; }
//       LogDebug("RawToDigi") << "[SiStripRawToDigiUnpacker::digiInfo] vr->data.size(): " << vr->data.size();
//       for ( uint16_t ivr = 0; ivr < vr->data.size(); ivr++ ) { anal_.vrDigi( ivr, vr->data[ivr].adc() ); }
//     } else { LogDebug("RawToDigi") << "[SiStripRawToDigiUnpacker::digiInfo] Key not found for VR digis"; } 
//     // PR
//     vector< edm::DetSet<SiStripRawDigi> >::const_iterator pr = proc_raw->find( *ikey );
//     if ( pr != proc_raw->end() ) {
//       if ( pr->data.empty() ) { LogDebug("RawToDigi") << "[SiStripRawToDigiUnpacker::digiInfo] No PR digis found!"; }
//       LogDebug("RawToDigi") << "[SiStripRawToDigiUnpacker::digiInfo] pr->data.size(): " << pr->data.size();
//       for ( uint16_t ipr = 0; ipr < pr->data.size(); ipr++ ) { anal_.prDigi( ipr, pr->data[ipr].adc() ); }
//     } else { LogDebug("RawToDigi") << "[SiStripRawToDigiUnpacker::digiInfo] Key not found for PR digis"; } 
//     // ZS
//     vector< edm::DetSet<SiStripDigi> >::const_iterator zs = zero_suppr->find( *ikey );
//     if ( zs != zero_suppr->end() ) {
//       if ( zs->data.empty() ) { LogDebug("RawToDigi") << "[SiStripRawToDigiUnpacker::digiInfo] No ZS digis found!"; }
//       LogDebug("RawToDigi") << "[SiStripRawToDigiUnpacker::digiInfo] zs->data.size(): " << zs->data.size();
//       for ( uint16_t izs = 0; izs < zs->data.size(); izs++ ) { anal_.zsDigi( zs->data[izs].strip(), zs->data[izs].adc() ); }
//     } else { LogDebug("RawToDigi") << "[SiStripRawToDigiUnpacker::digiInfo] Key not found for ZS digis"; } 
//   }
// }


