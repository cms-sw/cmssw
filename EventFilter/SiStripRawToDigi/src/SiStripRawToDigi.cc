#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToDigi.h"
// fwk, utilities
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Utilities/Timing/interface/TimingReport.h"
// data formats
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripEventSummary.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigis.h"
#include "DataFormats/SiStripDetId/interface/SiStripReadoutKey.h"
// cabling
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
//
#include "EventFilter/SiStripRawToDigi/interface/TFHeaderDescription.h"
#include "interface/shared/include/fed_header.h"
#include "interface/shared/include/fed_trailer.h"
// fed exception handling 
#include "ICException.hh"
// std
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace std;

// -----------------------------------------------------------------------------
/** */
SiStripRawToDigi::SiStripRawToDigi( int16_t header_bytes, 
				    int16_t dump_frequency, 
				    bool use_fed_key,
				    uint16_t trigger_fed_id ) : 
  fedEvent_(0), 
  fedDescription_(0),
  headerBytes_( header_bytes ),
  dumpFrequency_( dump_frequency ),
  useFedKey_( use_fed_key ),
  triggerFedId_( trigger_fed_id ),
  anal_("SiStripRawToDigi")
{
  edm::LogInfo("RawToDigi") << "[SiStripRawToDigi::SiStripRawToDigi] Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
SiStripRawToDigi::~SiStripRawToDigi() {
  edm::LogInfo("RawToDigi") << "[SiStripRawToDigi::~SiStripRawToDigi] Destructing object...";
  if ( fedEvent_ ) { delete fedEvent_; }
  if ( fedDescription_ ) { delete fedDescription_; }

}

// -----------------------------------------------------------------------------
/** 
    Input: FEDRawDataCollection. Output: StripDigiCollection.
    Retrieves and iterates through FED buffers, extract FEDRawData
    from collection and (optionally) dumps raw data to stdout, locates
    start of FED buffer by identifying DAQ header, creates new
    Fed9UEvent object using current FEDRawData buffer, dumps FED
    buffer to stdout, retrieves data from various header fields
*/
void SiStripRawToDigi::createDigis( const uint32_t& event,
				    edm::ESHandle<SiStripFedCabling>& cabling,
				    edm::Handle<FEDRawDataCollection>& buffers,
				    auto_ptr< edm::DetSetVector<SiStripRawDigi> >& scope_mode,
				    auto_ptr< edm::DetSetVector<SiStripRawDigi> >& virgin_raw,
				    auto_ptr< edm::DetSetVector<SiStripRawDigi> >& proc_raw,
				    auto_ptr< edm::DetSetVector<SiStripDigi> >& zero_suppr,
				    auto_ptr< SiStripEventSummary >& summary,
				    auto_ptr< SiStripDigis >& digis ) {

  cout << "[SiStripRawToDigi::createDigis] Event number: " << event << endl;

  // Debug info
  anal_.addEvent();
 
  // Extract Trigger FED information
  // triggerFedId_ = cabling->triggerFedId(); //@@ from cabling?!
  const FEDRawData& trigger_fed = buffers->FEDData( static_cast<int>(triggerFedId_) );
  if ( triggerFedId_ && trigger_fed.size() ) { triggerFed( trigger_fed, summary ); }
  //dumpRawData( triggerFedId_, trigger_fed ); 

  // Retrieve FED ids from cabling map and iterate through 
  uint32_t appended_bytes = 0;
  vector<uint16_t>::const_iterator ifed = cabling->feds().begin();
  for ( ; ifed != cabling->feds().end(); ifed++ ) {
    anal_.addFed();
    LogDebug("RawToDigi") << "[SiStripRawToDigi::createDigis] Extracting payload from FED id: " << *ifed;

    // Retrieve FED raw data for given FED 
    const FEDRawData& input = buffers->FEDData( static_cast<int>(*ifed) );
    //if ( dumpFrequency_ && !(event%dumpFrequency_) ) { dumpRawData( *ifed, input ); }
    
    // Locate start of FED buffer within raw data
    FEDRawData output; 
    locateStartOfFedBuffer( *ifed, input, output );
    appended_bytes = input.size() - output.size();
    Fed9U::u32* data_u32 = reinterpret_cast<Fed9U::u32*>( const_cast<unsigned char*>( output.data() ) );
    Fed9U::u32  size_u32 = static_cast<Fed9U::u32>( output.size() / 4 ); 

    if ( fedEvent_ ) { delete fedEvent_; fedEvent_ = 0; }
    fedEvent_ = new Fed9U::Fed9UEvent(); //@@ because of bug in fed sw?
    try {
      fedEvent_->Init( data_u32, fedDescription_, size_u32 ); 
      fedEvent_->checkEvent(); //@@ change checkEvent() so that it doesn't use getDaqMode()!
    } catch(...) { handleException( "SiStripRawToDigi::createDigis",
				    "Problem unpacking FED buffer" ); } 
    
    // Dump of FED buffer to stdout
    if ( dumpFrequency_ && !(event%dumpFrequency_) ) {
      stringstream ss;
      fedEvent_->dump( ss );
      LogDebug("RawToDigi") << ss.str();
    }
    
    // Retrieve DAQ/TK header information
    uint32_t run_type = fedEvent_->getEventType();
    uint32_t ev_num   = fedEvent_->getEventNumber();
    uint32_t bunchx   = fedEvent_->getBunchCrossing();
    uint32_t ev_type  = fedEvent_->getSpecialTrackerEventType();
    uint32_t daq_reg  = fedEvent_->getDaqRegister();
    if ( dumpFrequency_ && !(event%dumpFrequency_) ) {
      stringstream ss;
      ss << "[SiStripRawToDigi::createDigis]"
	 << "  Run Type: " << run_type 
	 << "  Event Number: " << ev_num 
	 << "  Bunch Crossing: " << bunchx 
	 << "  FED Readout Mode: " << ev_type 
	 << "  DAQ Register: " << daq_reg; 
      LogDebug("RawToDigi") << ss.str();
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
	handleException( "SiStripRawToDigi::createDigis",
			 "Problem unpacking FED payload" ); 
	continue; //@@ is this ok?
      } 
      
      // Retrieve cabling map information and define "FED key" for Digis
      const FedChannelConnection& conn = cabling->connection( *ifed, chan );
      
      // Determine whether DetId or FED key should be used to index digi containers
      uint32_t fed_key = SiStripReadoutKey::key( conn.fedId(), conn.fedCh() );
      uint32_t key     = (useFedKey_ || ev_type==1) ? fed_key : conn.detId();
      uint16_t ipair   = (useFedKey_ || ev_type==1) ? 0 : conn.apvPairNumber();
//       stringstream ss; 
//       ss << "[SiStripRawToDigi::createDigis]" 
// 	 << "  FED id/ch/key: " 
// 	 << conn.fedId() << "/" << conn.fedCh() << "/" 
// 	 << hex << setfill('0') << setw(8) << fed_key << dec
// 	 << " UseFedKey?/ScopeMode?/DetId/ApvPairNumber/key/ipair: " 
// 	 << useFedKey_ << "/" << (ev_type==1) << "/" << conn.detId() 
// 	 << "/" << conn.apvPairNumber() << "/" 
// 	 << hex << setfill('0') << setw(8) << key << dec 
// 	 << "/" << ipair;
//       LogDebug("RawToDigi") << ss.str();
      
      // Check for non-zero key OR scope mode
      if ( !key ) { continue; }
      anal_.addChan();
      
      if ( ev_type == 1 ) { // SCOPE MODE
	edm::DetSet<SiStripRawDigi>& sm = scope_mode->find_or_insert( key );
	vector<uint16_t> samples; samples.reserve( 1024 ); // theoretical maximum
	samples = fedEvent_->feUnit( iunit ).channel( ichan ).getSamples();
	if ( samples.empty() ) { 
	  edm::LogWarning("Commissioning") << "[SiStripRawToDigi::createDigis] No SM digis found!"; 
	} else {
	  sm.data.clear(); sm.data.reserve( samples.size() ); sm.data.resize( samples.size() ); 
	  for ( uint16_t i = 0; i < samples.size(); i++ ) {
	    sm.data[i] = SiStripRawDigi( samples[i] ); 
	    anal_.smDigi( i, sm.data[i].adc() );
	  }
	  LogDebug("Commissioning") << "Extracted " << samples.size() 
				    << " SCOPE MODE digis (samples[0] = " << samples[0] 
				    << ") from FED id/ch " 
				    << conn.fedId() << "/" << conn.fedCh();
	}
      } else if ( ev_type == 2 ) { // VIRGIN RAW
	edm::DetSet<SiStripRawDigi>& vr = virgin_raw->find_or_insert( key );
	vector<uint16_t> samples; samples.reserve(256);
	samples = fedEvent_->channel( iunit, ichan ).getSamples();
	if ( samples.empty() ) { 
	  edm::LogWarning("Commissioning") << "[SiStripRawToDigi::createDigis] No VR digis found!"; 
	} else {
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
	    anal_.vrDigi( ipair*256+i, vr.data[ipair*256+i].adc() );
	  }
	  LogDebug("Commissioning") << "Extracted " << samples.size() 
				    << " VIRGIN RAW digis (samples[0] = " << samples[0] 
				    << ") from FED id/ch " 
				    << conn.fedId() << "/" << conn.fedCh();
	}
      } else if ( ev_type == 6 ) { // PROCESSED RAW
	edm::DetSet<SiStripRawDigi>& pr = proc_raw->find_or_insert( key ) ;
	vector<uint16_t> samples; samples.reserve(256);
	samples = fedEvent_->channel( iunit, ichan ).getSamples();
	if ( samples.empty() ) { 
	  edm::LogWarning("Commissioning") << "[SiStripRawToDigi::createDigis] No PR digis found!"; 
	} else {
	  if ( pr.data.size() < static_cast<uint16_t>(256*(ipair+1)) ) { 
	    pr.data.reserve( 256*(ipair+1) ); pr.data.resize( 256*(ipair+1) ); 
	  }
	  int physical;
	  for ( uint16_t i = 0; i < samples.size(); i++ ) {
	    physical = i%128; 
	    (i/128) ? physical=physical*2+1 : physical=physical*2; // multiplexed data
	    pr.data[ipair*256+i] = SiStripRawDigi( samples[physical] ); 
	    anal_.prDigi( ipair*256+i, pr.data[ipair*256+i].adc() );
	  } 
	  LogDebug("Commissioning") << "Extracted " << samples.size() 
				    << " PROCESSED RAW digis (samples[0] = " << samples[0] 
				    << ") from FED id/ch " 
				    << conn.fedId() << "/" << conn.fedCh();
	}
      } else if ( ev_type == 10 ) { // ZERO SUPPRESSED
	edm::DetSet<SiStripDigi>& zs = zero_suppr->find_or_insert( key );
	zs.data.reserve(256); // theoretical maximum (768/3, ie, clusters separated by at least 2 strips)
	Fed9U::Fed9UEventIterator fed_iter = const_cast<Fed9U::Fed9UEventChannel&>(fedEvent_->channel( iunit, ichan )).getIterator();
	for (Fed9U::Fed9UEventIterator i = fed_iter+7; i.size() > 0; /**/) {
	  unsigned char first_strip = *i++; // first strip of cluster
	  unsigned char width = *i++;       // cluster width in strips 
	  for ( uint16_t istr = 0; istr < width; istr++) {
	    uint16_t strip = ipair*256 + first_strip + istr;
	    zs.data.push_back( SiStripDigi( strip, static_cast<uint16_t>(*i) ) );
	    anal_.zsDigi( zs.data.back().strip(), zs.data.back().strip() );
	    *i++; // Iterate to next sample
	  }
	  LogDebug("Commissioning") << "Extracted " << zs.data.size() 
				    << " PROCESSED RAW digis (samples[0] = " << zs.data.front().adc()
				    << ") from FED id/ch " 
				    << conn.fedId() << "/" << conn.fedCh();
	}
	//sort( zs.data.begin(), zs.data.end() ); //@@ necessary?
      } else { // UNKNOWN READOUT MODE => ASSUME SCOPE MODE
	stringstream ss;
	ss << "[SiStripRawToDigi::createDigis]"
	   << " Unknown FED readout mode (ev_type = " << ev_type << ")!" 
	   << " Assuming SCOPE MODE..."; 
	edm::LogError("RawToDigi") << ss.str();
	edm::DetSet<SiStripRawDigi>& sm = scope_mode->find_or_insert( key );
	vector<uint16_t> samples; samples.reserve( 1024 ); // theoretical maximum
	samples = fedEvent_->feUnit( iunit ).channel( ichan ).getSamples();
	if ( samples.empty() ) { 
	  edm::LogWarning("Commissioning") << "[SiStripRawToDigi::createDigis] No SM digis found!"; 
	} else {
	  sm.data.clear(); sm.data.reserve( samples.size() ); sm.data.resize( samples.size() ); 
	  for ( uint16_t i = 0; i < samples.size(); i++ ) {
	    sm.data[i] = SiStripRawDigi( samples[i] ); 
	    anal_.smDigi( i, sm.data[i].adc() );
	  }
	  LogDebug("Commissioning") << "Extracted " << samples.size() 
				    << " SCOPE MODE digis (samples[0] = " << samples[0] 
				    << ") from FED id/ch " 
				    << conn.fedId() << "/" << conn.fedCh();
	}
      }
    }
  }

  // Create SiStripDigis object
  digis = auto_ptr<SiStripDigis>( new SiStripDigis( buffers, cabling->feds(), appended_bytes ) );
    
  if ( fedEvent_ ) { 
    delete fedEvent_; //@@ because of bug in fed sw
    fedEvent_=0; 
  }
  
}

// -----------------------------------------------------------------------------
/** */
void SiStripRawToDigi::triggerFed( const FEDRawData& trigger_fed,
				   auto_ptr< SiStripEventSummary >& summary ) {
  static const string method = "SiStripRawToDigi::triggerFed"; 

  if ( !trigger_fed.data() ) { 
    stringstream ss;
    ss << "["<<method<<"]"
       << " NULL pointer to buffer for FED id " << triggerFedId_ << "!";
    edm::LogError("SiStripRawToDigi") << ss.str();
    //throw cms::Exception("SiStripRawToDigi") << ss.str();
    return;
  }
  
  if ( trigger_fed.size() < sizeof(fedh_t)  ) {
    stringstream ss;
    ss << "["<<method<<"]"
       << " Buffer for FED id " << triggerFedId_ 
       << " has size " << trigger_fed.size() << "!";
    edm::LogError("SiStripRawToDigi") << ss.str();
    //throw cms::Exception("SiStripRawToDigi") << ss.str();
    return;
  }
  
  // Recast pointers and buffer size
  uint8_t*  temp = const_cast<uint8_t*>( trigger_fed.data() );
  uint32_t* data_u32 = reinterpret_cast<uint32_t*>( temp ) + sizeof(fedh_t)/sizeof(uint32_t) + 1;
  uint32_t  size_u32 = trigger_fed.size()/sizeof(uint32_t) - sizeof(fedh_t)/sizeof(uint32_t) - 1;
  
  // Check whether buffer is really "trigger FED" (and not a FED buffer)
  fedh_t* fed_header  = reinterpret_cast<fedh_t*>( temp );
  fedt_t* fed_trailer = reinterpret_cast<fedt_t*>( temp + trigger_fed.size() - sizeof(fedt_t) );
  if ( fed_trailer->conscheck != 0xDEADFACE ) {
    stringstream ss;
    ss << "["<<method<<"]"
       << " Buffer does not appear to contain 'Trigger FED' information!"
       << " Trigger FED id: " << triggerFedId_
       << " Source id: " << hex <<setw(8) << setfill('0') << "0x" << fed_header->sourceid << dec;
    edm::LogError("SiStripRawToDigi") << ss.str();
    //throw cms::Exception("SiStripRawToDigi") << ss.str();
    return;
  } //@@ if not trigger FED, perform search?...
  
  if ( size_u32 > sizeof(TFHeaderDescription)/sizeof(uint32_t) ) {
    
    TFHeaderDescription* header = (TFHeaderDescription*) data_u32;
    stringstream ss;
    ss << "["<<method<<"]"
       << "  getBunchCrossing: " << header->getBunchCrossing()
       << "  getNumberOfChannels: " << header->getNumberOfChannels() 
       << "  getNumberOfSamples: " << header->getNumberOfSamples()
       << "  getFedType: 0x" 
       << hex << setw(8) << setfill('0') << header->getFedType() << dec
       << "  getFedId: " << header->getFedId()
       << "  getFedEventNumber: " << header->getFedEventNumber();
    LogDebug("RawToDigi") << ss.str();
      
    // Write event-specific data to event
    summary->event( static_cast<uint32_t>( header->getFedEventNumber()) );
    summary->bx( static_cast<uint32_t>( header->getBunchCrossing()) );
      
    // Write commissioning information to event 
    uint32_t hsize = sizeof(TFHeaderDescription)/sizeof(uint32_t);
    uint32_t* head = &data_u32[hsize];
    //summary->commissioningInfo( head );
      
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
void SiStripRawToDigi::locateStartOfFedBuffer( uint16_t fed_id,
					       const FEDRawData& input,
					       FEDRawData& output ) {
  
  // Check size of input buffer
  if ( input.size() < 24 ) { 
    output.resize( input.size() ); // Return UNadjusted buffer start position and size
    memcpy( output.data(), input.data(), input.size() );
    stringstream ss; 
    ss << "[SiStripRawToDigi::locateStartOfFedBuffer] "
       << "Input FEDRawData with FED id " << fed_id 
       << " has size " << input.size() << "\n";
    edm::LogError("SiStripRawToDigi") << ss.str();
    throw cms::Exception("SiStripRawToDigi") << ss.str();
    return;
  } 
  
  // Iterator through buffer to find DAQ header 
  bool found = false;
  uint16_t ichar = 0;
  while ( ichar < input.size()-16 && !found ) {
    uint16_t offset = headerBytes_ < 0 ? ichar : headerBytes_; // Negative value means use "search mode" to find DAQ header
    uint32_t* input_u32   = reinterpret_cast<uint32_t*>( const_cast<unsigned char*>( input.data() ) + offset );
    uint32_t* fed_trailer = reinterpret_cast<uint32_t*>( const_cast<unsigned char*>( input.data() ) + input.size() - 8 );
//     cout << "FED trailer" 
// 	 << ":" << hex << setfill('0') << setw(0) << fed_trailer[0] << dec
// 	 << ":" << hex << setfill('0') << setw(0) << fed_trailer[1] << dec
// 	 << ":" << hex << setfill('0') << setw(0) << ((fed_trailer[0] & 0x00FFFFFF) * 0x8) << dec
// 	 << ":" << hex << setfill('0') << setw(0) << ((fed_trailer[1] & 0x00FFFFFF) * 0x8) << dec
// 	 << ":" << (input.size() - offset) 
// 	 << ":" << (((fed_trailer[0] & 0x00FFFFFF) * 0x8) == (input.size() - offset)) 
// 	 << ":" << (((fed_trailer[1] & 0x00FFFFFF) * 0x8) == (input.size() - offset)) << endl;
    if ( (input_u32[0]    & 0xF0000000) == 0x50000000 &&
	 //(input_u32[1]    & 0x0000000F) == 0x00000008 && 
	 //(input_u32[2]    & 0xFF000000) == 0xED000000 &&
	 (fed_trailer[0]  & 0xF0000000) == 0xA0000000 && 
	 ((fed_trailer[0] & 0x00FFFFFF) * 0x8) == (input.size() - offset) ) {
      // Found DAQ header at byte position 'offset'
      found = true;
      output.resize( input.size()-offset );
      memcpy( output.data(),         // target
	      input.data()+offset,   // source
	      input.size()-offset ); // nbytes
      if ( headerBytes_ < 0 ) {
	edm::LogInfo("RawToDigi") << "[SiStripRawToDigi::locateStartOfFedBuffer]" 
				  << " FED buffer has been found at byte position " 
				  << offset << " with a size of " << input.size()-offset << " bytes";
	edm::LogInfo("RawToDigi") << "[SiStripRawToDigi::locateStartOfFedBuffer]" 
				  << " Adjust the configurable 'AppendedHeaderBytes' to " << offset;
      }
    } else if ( (input_u32[1]    & 0xF0000000) == 0x50000000 &&
		//(input_u32[0]    & 0x0000000F) == 0x00000008 && 
		//(input_u32[3]    & 0xFF000000) == 0xED000000 &&
		(fed_trailer[1]  & 0xF0000000) == 0xA0000000 &&
		((fed_trailer[1] & 0x00FFFFFF) * 0x8) == (input.size() - offset) ) {
      // Found DAQ header (with MSB and LSB 32-bit words swapped) at byte position 'offset' 
      found = true;
      output.resize( input.size()-offset );
      uint32_t* output_u32 = reinterpret_cast<uint32_t*>( const_cast<unsigned char*>( output.data() ) + offset );
      uint16_t iter = 0; 
      while ( iter < input.size() / sizeof(uint32_t) ) {
	output_u32[iter] = input_u32[iter+1];
	output_u32[iter+1] = input_u32[iter];
	iter+=2;
      }
      if ( headerBytes_ < 0 ) {
	edm::LogInfo("RawToDigi") << "[SiStripRawToDigi::locateStartOfFedBuffer]" 
				  << " FED buffer (with MSB and LSB 32-bit words swapped) has been found at byte position " 
				  << offset << " with a size of " << input.size()-offset << " bytes";
	edm::LogInfo("RawToDigi") << "[SiStripRawToDigi::locateStartOfFedBuffer]" 
				  << " Adjust the configurable 'AppendedHeaderBytes' to " << offset;
      }
    } else { headerBytes_ < 0 ? found = false : found = true; }
    ichar++;
  }      
  
  // Check size of output buffer
  if ( output.size() == 0 ) { // Did not find DAQ header after search. 
    output.resize( input.size() ); // Return UNadjusted buffer start position and size
    memcpy( output.data(), input.data(), input.size() );
    stringstream ss;
    if ( headerBytes_ < 0 ) {
      ss << "[SiStripRawToDigi::locateStartOfFedBuffer]"
	 << " DAQ header not found within data buffer!";
    } else {
      uint32_t* input_u32 = reinterpret_cast<uint32_t*>( const_cast<unsigned char*>( input.data() ) );
      ss << "[SiStripRawToDigi::locateStartOfFedBuffer]"
	 << " DAQ header not found at expected location!"
	 << " First 64-bit word of buffer is 0x"
	 << hex 
	 << setfill('0') << setw(8) << input_u32[0] 
	 << setfill('0') << setw(8) << input_u32[1] 
	 << dec
	 << ". Adjust 'AppendedHeaderBytes' configurable"
	 << " to negative value to activate 'search mode'";
    }
    edm::LogError("SiStripRawToDigi") << ss.str();
    throw cms::Exception("SiStripRawToDigi") << ss.str();
  } else if ( output.size() < 24 ) { // Found DAQ header after search, but too few words
    stringstream ss; 
    ss << "[SiStripRawToDigi::locateStartOfFedBuffer]"
       << " Unexpected buffer size! FEDRawData with FED id " << fed_id 
       << " has size " << output.size();
    edm::LogError("RawToDigi") << ss.str();
    throw cms::Exception("SiStripRawToDigi") << ss.str();
  } 
  
}

//------------------------------------------------------------------------------
/** 
    Dumps raw data to stdout (NB: payload is byte-swapped,
    headers/trailer are not).
*/
void SiStripRawToDigi::dumpRawData( uint16_t fed_id, 
				    const FEDRawData& buffer,
				    ostream& os ) {
  //@@ need to pipe info to ostream!!
  LogDebug("RawToDigi") << "[SiStripRawToDigi::dumpRawData] "
			<< "Dump of buffer from FED id " <<  fed_id 
			<< " which contains " << buffer.size() <<" bytes";
  LogDebug("RawToDigi") << "NB: payload is byte-swapped and empty words have been removed.";
  uint32_t* buffer_u32 = reinterpret_cast<uint32_t*>( const_cast<unsigned char*>( buffer.data() ) );
  unsigned int empty = 0;
  for ( uint32_t i = 0; i < buffer.size()/8; i++ ) {
    unsigned int temp0 = buffer_u32[i*2] & 0xFFFFFFFF;
    unsigned int temp1 = buffer_u32[i*2+1] & 0xFFFFFFFF;
    if ( !temp0 && !temp1 ) { empty++; }
    else { 
      if ( empty ) { 
	LogDebug("RawToDigi") << "       [ empty  words ]"; 
	empty = 0; 
      }
      stringstream ss;
      ss << setw(5) << dec << i*8 << ": " << hex 
	 << setfill('0') << setw(8) << temp0 
	 << setfill('0') << setw(8) << temp1 
	 << dec;
      LogDebug("RawToDigi") << ss.str();
    }
  }
  LogDebug("RawToDigi") << "[SiStripRawToDigi::dumpRawData] "
			<< "End of FED buffer";
}

// -----------------------------------------------------------------------------
// 
void SiStripRawToDigi::handleException( const string& method_name,
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
    edm::LogError("SiStripRawToDigi") << ss.str();
    throw cms::Exception("SiStripRawToDigi") << ss.str();
  }
  catch ( const exception& e ) {
    stringstream ss;
    ss << "Caught std::exception in ["
       << method_name << "] with message: \n" 
       << e.what();
    if ( extra_info != "" ) { ss << "Additional info: " << extra_info; }
    edm::LogError("SiStripRawToDigi") << ss.str();
    throw cms::Exception("SiStripRawToDigi") << ss.str();
  }
  catch (...) {
    stringstream ss;
    ss << "Caught unknown exception in ["
       << method_name << "]";
    if ( extra_info != "" ) { ss << "\n" << "Additional info: " << extra_info; }
    edm::LogError("SiStripRawToDigi") << ss.str();
    throw cms::Exception("SiStripRawToDigi") << ss.str();
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
// 	edm::LogInfo("RawToDigi") << "[SiStripRawToDigi::locateStartOfFedBuffer]" 
// 				  << " FED buffer has been found at byte position " 
// 				  << ichar << " with a size of " << input.size()-ichar << " bytes";
// 	edm::LogInfo("RawToDigi") << "[SiStripRawToDigi::locateStartOfFedBuffer]" 
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
//       ss << "[SiStripRawToDigi::locateStartOfFedBuffer]"
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
//     ss << "[SiStripRawToDigi::locateStartOfFedBuffer]"
//        << " Unexpected buffer size! FEDRawData with FED id " << fed_id 
//        << " has size " << output.size();
//     edm::LogError("RawToDigi") << ss.str();
//   } 
 









// // -----------------------------------------------------------------------------
// /** */
// void SiStripRawToDigi::digiInfo( vector<uint32_t>& keys, //@@ TEMP!
// 				 auto_ptr< edm::DetSetVector<SiStripRawDigi> >& scope_mode,
// 				 auto_ptr< edm::DetSetVector<SiStripRawDigi> >& virgin_raw,
// 				 auto_ptr< edm::DetSetVector<SiStripRawDigi> >& proc_raw,
// 				 auto_ptr< edm::DetSetVector<SiStripDigi> >& zero_suppr ) {
//   LogDebug("RawToDigi") << "[SiStripRawToDigi::digiInfo] Number of keys: " << keys.size();
//   vector<uint32_t>::iterator ikey;
//   for ( ikey = keys.begin(); ikey != keys.end(); ikey++ ) {
//     LogDebug("RawToDigi") << "[SiStripRawToDigi::digiInfo] Key: " << hex << setfill('0') << setw(8) << *ikey << dec;
//     // SM
//     vector< edm::DetSet<SiStripRawDigi> >::const_iterator sm = scope_mode->find( *ikey );
//     if ( sm != scope_mode->end() ) {
//       if ( sm->data.empty() ) { LogDebug("RawToDigi") << "[SiStripRawToDigi::digiInfo] No SM digis found!"; }
//       LogDebug("RawToDigi") << "[SiStripRawToDigi::digiInfo] sm->data.size(): " << sm->data.size();
//       for ( uint16_t ism = 0; ism < sm->data.size(); ism++ ) { anal_.smDigi( ism, sm->data[ism].adc() ); }
//     } else { LogDebug("RawToDigi") << "[SiStripRawToDigi::digiInfo] Key not found for SM digis"; }
//     // VR
//     vector< edm::DetSet<SiStripRawDigi> >::const_iterator vr = virgin_raw->find( *ikey );
//     if ( vr != virgin_raw->end() ) {
//       if ( vr->data.empty() ) { LogDebug("RawToDigi") << "[SiStripRawToDigi::digiInfo] No VR digis found!"; }
//       LogDebug("RawToDigi") << "[SiStripRawToDigi::digiInfo] vr->data.size(): " << vr->data.size();
//       for ( uint16_t ivr = 0; ivr < vr->data.size(); ivr++ ) { anal_.vrDigi( ivr, vr->data[ivr].adc() ); }
//     } else { LogDebug("RawToDigi") << "[SiStripRawToDigi::digiInfo] Key not found for VR digis"; } 
//     // PR
//     vector< edm::DetSet<SiStripRawDigi> >::const_iterator pr = proc_raw->find( *ikey );
//     if ( pr != proc_raw->end() ) {
//       if ( pr->data.empty() ) { LogDebug("RawToDigi") << "[SiStripRawToDigi::digiInfo] No PR digis found!"; }
//       LogDebug("RawToDigi") << "[SiStripRawToDigi::digiInfo] pr->data.size(): " << pr->data.size();
//       for ( uint16_t ipr = 0; ipr < pr->data.size(); ipr++ ) { anal_.prDigi( ipr, pr->data[ipr].adc() ); }
//     } else { LogDebug("RawToDigi") << "[SiStripRawToDigi::digiInfo] Key not found for PR digis"; } 
//     // ZS
//     vector< edm::DetSet<SiStripDigi> >::const_iterator zs = zero_suppr->find( *ikey );
//     if ( zs != zero_suppr->end() ) {
//       if ( zs->data.empty() ) { LogDebug("RawToDigi") << "[SiStripRawToDigi::digiInfo] No ZS digis found!"; }
//       LogDebug("RawToDigi") << "[SiStripRawToDigi::digiInfo] zs->data.size(): " << zs->data.size();
//       for ( uint16_t izs = 0; izs < zs->data.size(); izs++ ) { anal_.zsDigi( zs->data[izs].strip(), zs->data[izs].adc() ); }
//     } else { LogDebug("RawToDigi") << "[SiStripRawToDigi::digiInfo] Key not found for ZS digis"; } 
//   }
// }


