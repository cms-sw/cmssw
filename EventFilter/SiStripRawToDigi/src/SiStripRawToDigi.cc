#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToDigi.h"
// timing
#include "Utilities/Timing/interface/TimingReport.h"
// data formats
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripEventSummary.h"
// cabling
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
// fed exception handling 
#include "ICException.hh"
// std
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
/** */
SiStripRawToDigi::SiStripRawToDigi( int16_t header_bytes, 
				    int16_t dump_frequency, 
				    bool use_det_id,
				    uint16_t trigger_fed_id ) : 
  fedEvent_(0), 
  fedDescription_(0),
  headerBytes_( header_bytes ),
  dumpFrequency_( dump_frequency ),
  useDetId_( use_det_id ),
  triggerFedId_( trigger_fed_id ),
  nFeds_(0), 
  nDets_(0), 
  nDigis_(0),
  position_(), 
  landau_()
{
  cout << "[SiStripRawToDigi::SiStripRawToDigi] " 
       << "Constructing object..." << endl;
  
  landau_.clear(); landau_.reserve(1024); landau_.resize(1024,0);
  position_.clear(); position_.reserve(768); position_.resize(768,0);
  
}

// -----------------------------------------------------------------------------
/** */
SiStripRawToDigi::~SiStripRawToDigi() {
  cout << "[SiStripRawToDigi::~SiStripRawToDigi] " 
       << "Destructing object..." << endl;

  if ( 0 ) {
    cout << "[SiStripRawToDigi::~SiStripRawToDigi]"
	 << " Some cumulative counters:"
	 << " Number of FEDs " << nFeds_ 
	 << ", number of Dets " << nDets_ 
	 << ", number of Digis " << nDigis_ << endl;
    cout << "[SiStripRawToDigi::~SiStripRawToDigi]"
	 << " Digi statistics (vs strip position): " << endl;
    int tmp1 = 0;
    for (unsigned int i=0; i<position_.size(); i++) {
      if ( i<10 ) { cout << "Strip: " << i 
			 << ",  Digis: " << position_[i] 
			 << endl; }
      tmp1 += position_[i];
    }
    cout << "Ndigis " << tmp1 << endl << endl;
    cout << "[SiStripRawToDigi::~SiStripRawToDigi]"
	 << " Landau statistics: " << endl;
    int tmp2 = 0;
    for (unsigned int i=0; i<landau_.size(); i++) {
      if ( i<10 ) { cout << "ADC: " << i 
			 << ",  Digis: " << landau_[i] 
			 << endl; }
      tmp2 += landau_[i];
    }
    cout << "Ndigis " << tmp2 << endl << endl;
  }
  position_.clear();
  landau_.clear();
  
  if ( fedEvent_ ) { 
    delete fedEvent_; 
    fedEvent_ = 0; 
  }
  if ( fedDescription_ ) { 
    delete fedDescription_; 
    fedDescription_ = 0; 
  }
  
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
void SiStripRawToDigi::createDigis( edm::ESHandle<SiStripFedCabling>& cabling,
				    edm::Handle<FEDRawDataCollection>& buffers,
				    auto_ptr< edm::DetSetVector<SiStripRawDigi> >& scope_mode,
				    auto_ptr< edm::DetSetVector<SiStripRawDigi> >& virgin_raw,
				    auto_ptr< edm::DetSetVector<SiStripRawDigi> >& proc_raw,
				    auto_ptr< edm::DetSetVector<SiStripDigi> >& zero_suppr,
				    auto_ptr< SiStripEventSummary >& summary ) {
  cout << "[SiStripRawToDigi::createDigis]" << endl; 
  
  triggerFed( buffers, summary ); // Extract Trigger FED information

  // Retrieve FED ids from cabling map and iterate through 
  const vector<uint16_t>& fed_ids = cabling->feds(); 
  vector<uint16_t>::const_iterator ifed;
  for ( ifed = fed_ids.begin(); ifed != fed_ids.end(); ifed++ ) {
    
    cout << "[SiStripRawToDigi::createDigis]"
	 << " extracting payload from FED id: " << *ifed << endl; 
    nFeds_++;

    // Retrieve FED raw data for given FED 
    const FEDRawData& input = buffers->FEDData( static_cast<int>(*ifed) );
    dumpRawData( *ifed, input );

    // Locate start of FED buffer within raw data
    FEDRawData output; 
    locateStartOfFedBuffer( *ifed, input, output );
    Fed9U::u32* data_u32 = reinterpret_cast<Fed9U::u32*>( const_cast<unsigned char*>( output.data() ) );
    Fed9U::u32  size_u32 = static_cast<Fed9U::u32>( output.size() / 4 ); 
    
    if ( fedEvent_ ) { delete fedEvent_; fedEvent_ = 0; }
    fedEvent_ = new Fed9U::Fed9UEvent(); //@@ still necessary?

    // Initialise FED9UEvent object
    try {
      fedEvent_->Init( data_u32, fedDescription_, size_u32 ); 
    } 
    catch( ICUtils::ICException& e ) {
      stringstream ss;
      ss << "[SiStripRawToDigi::createDigis]"
	 << " Caught ICExeption: " << e.what() << endl;
      throw string( ss.str() );
      cout << ss.str() << endl; 
    } 
    catch(...) {
      stringstream ss;
      ss << "[SiStripRawToDigi::createDigis]"
	 << " Unknown exception thrown by Fed9UEvent!" << endl;
      throw string( ss.str() );
      cout << ss.str() << endl; 
    }
    
    // Dump of FED buffer to stdout
    if ( 0 ) { 
      stringstream ss;
      fedEvent_->dump( ss );
      cout << ss.str() << endl;
    }

    // Retrieve DAQ/TK header information
    uint32_t run_type = fedEvent_->getEventType();
    uint32_t ev_num   = fedEvent_->getEventNumber();
    uint32_t bunchx   = fedEvent_->getBunchCrossing();
    uint32_t ev_type  = fedEvent_->getSpecialTrackerEventType();
    uint32_t daq_reg  = fedEvent_->getDaqRegister();
    cout << "[SiStripRawToDigi::createDigis]"
	 << " Run Type: " << run_type 
	 << ", Event Number: " << ev_num 
	 << ", FED Readout Mode: " << ev_type 
	 << ", Bunch Crossing: " << bunchx 
	 << ", DAQ Register: " << daq_reg << endl; 
    
    // Iterate through FED channels, extract payload and create Digis
    for ( uint16_t iunit = 0; iunit < fedEvent_->feUnits(); iunit++ ) {
      for ( uint16_t ichan = 0; ichan < fedEvent_->feUnit(iunit).channels(); ichan++ ) {
	uint16_t channel = iunit*12 + ichan;
	
	// Retrieve cabling map information and define "FED key" for Digis
	const FedChannelConnection& conn = cabling->connection( *ifed, channel );
	uint32_t fed_key = ( (*ifed)<<16 & 0xFFFF0000 ) & ( channel & 0x0000FFFF );
	uint16_t ipair = useDetId_ ? conn.pairId() : 0;
	
 	//@@ Check whether FED key already exists in DetSetVector?

	// Retrieve Digi containers (or create them if they don't exist)
	edm::DetSet<SiStripRawDigi>& sm = scope_mode->find_or_insert( fed_key );
	edm::DetSet<SiStripRawDigi>& vr = virgin_raw->find_or_insert( useDetId_ ? fed_key : conn.detId() );
	edm::DetSet<SiStripRawDigi>& pr = proc_raw->find_or_insert( useDetId_ ? fed_key : conn.detId() );
	edm::DetSet<SiStripDigi>&    zs = zero_suppr->find_or_insert( useDetId_ ? fed_key : conn.detId() );
		
	if ( ev_type == 1 ) { // SCOPE MODE
	  vector<uint16_t> samples = fedEvent_->channel( iunit, ichan ).getSamples();
	  sm.data.reserve( samples.size() ); sm.data.clear();
	  for ( uint16_t i = 0; i < samples.size(); i++ ) {
	    sm.data[i] = SiStripRawDigi( samples[i] ); 
	  }
	} else if ( ev_type == 2 ) { // VIRGIN RAW
	  vector<uint16_t> samples = fedEvent_->channel( iunit, ichan ).getSamples();
	  if ( vr.data.size() < static_cast<uint16_t>(256*(ipair+1)) ) { vr.data.resize( 256*(ipair+1) ); }
	  uint16_t physical;
	  uint16_t readout; 
	  for ( uint16_t i = 0; i < samples.size(); i++ ) {
	    physical = i%128;
	    readoutOrder( physical, readout ); // convert from physical to readout order
	    (i/128) ? readout=readout*2+1 : readout=readout*2; // multiplexed data
	    vr.data[ipair*256+i] = SiStripRawDigi( samples[readout] ); 
	  }
	} else if ( ev_type == 6 ) { // PROCESSED RAW
	  vector<uint16_t> samples = fedEvent_->channel( iunit, ichan ).getSamples();
	  if ( pr.data.size() < static_cast<uint16_t>(256*(ipair+1)) ) { pr.data.resize( 256*(ipair+1) ); }
	  int physical;
	  for ( uint16_t i = 0; i < samples.size(); i++ ) {
	    physical = i%128; 
	    (i/128) ? physical=physical*2+1 : physical=physical*2; // multiplexed data
	    pr.data[ipair*256+i] = SiStripRawDigi( samples[physical] ); 
	  } 
	} else if ( ev_type == 10 ) { // ZERO SUPPRESSED
	  Fed9U::Fed9UEventIterator fed_iter = const_cast<Fed9U::Fed9UEventChannel&>(fedEvent_->channel( iunit, ichan )).getIterator();
	  for (Fed9U::Fed9UEventIterator i = fed_iter+7; i.size() > 0; /**/) {
	    unsigned char first_strip = *i++; // first strip of cluster
	    unsigned char width = *i++;       // cluster width in strips 
	    for (uint16_t istr = 0; istr < width; istr++) {
	      uint16_t strip = ipair*256 + first_strip + istr;
	      zs.data.push_back( SiStripDigi( strip, static_cast<uint16_t>(*i) ) );
	      *i++; // Iterate to next sample
	    }
	  }
	  sort( zs.data.begin(), zs.data.end() ); //@@ necessary?
	} else { // UNKNOWN READOUT MODE (=> ASSUME SCOPE MODE)
	  stringstream ss;
	  ss << "[SiStripRawToDigi::createDigis]"
	     << " Unknown FED readout mode (ev_type)!" 
	     << " Assuming SCOPE MODE..." 
	     << ev_type << endl; 
	  throw string( ss.str() ); 
	  vector<uint16_t> samples = fedEvent_->channel( iunit, ichan ).getSamples();
	  sm.data.reserve( samples.size() ); sm.data.clear();
	  for ( uint16_t i = 0; i < samples.size(); i++ ) {
	    sm.data[i] = SiStripRawDigi( samples[i] ); 
	  }
	}
	
      }

    }
    
    if ( fedEvent_ ) { 
      delete fedEvent_; //@@ <- still necessary???
      fedEvent_=0; 
    }
    
  } 
  
  recordDebugInfo( scope_mode,
		   virgin_raw,
		   proc_raw,
		   zero_suppr );
  
}

// -----------------------------------------------------------------------------
/** */
void SiStripRawToDigi::triggerFed( edm::Handle<FEDRawDataCollection>& buffers,
				   auto_ptr< SiStripEventSummary >& summary ) {
  cout << "[SiStripRawToDigi::extractTriggerFedInfo]" << endl; 
  
  const FEDRawData& trigger_fed = buffers->FEDData( triggerFedId_ );
  
  const unsigned char* data = trigger_fed.data();
  unsigned int size = trigger_fed.size();
  
  if ( data && size ) {
    FEDHeader hdr(data);
    if ( !hdr.check() ) {
      cerr << "[SiStripRawToDigi::extractTriggerFedInfo]"
	   << " Problem with DAQ header for Trigger FED!" << endl; 
    }
    //     int triggerType();
    //     int lvl1ID();
    //     int bxID();
    //     int sourceID();
    //     int version();
    data += sizeof(FEDHeader);
    //@@ extract trigger fed data here.
  } else {
    cerr << "[SiStripRawToDigi::extractTriggerFedInfo]"
	 << " Problems retrieving TriggerFed data!"
	 << " data (ptr): " << hex << data << dec
	 << " size: " << size << endl;
  }

}

//------------------------------------------------------------------------------
/** 
    Remove any data appended prior to FED buffer
*/
void SiStripRawToDigi::locateStartOfFedBuffer( uint16_t fed_id,
					       const FEDRawData& input,
					       FEDRawData& output ) {
  cout << "[SiStripRawToDigi::locateStartOfFedBuffer]" << endl; 
  
  // Check on size of buffer
  if ( input.size() < 24 ) { 
    stringstream ss; 
    ss << "[SiStripRawToDigi::locateStartOfFedBuffer] "
       << "Input FEDRawData with FED id " << fed_id 
       << " has size " << input.size() << endl;
    throw string( ss.str() );
  } 
  
  unsigned long BOE1; // DAQ header, 4 MSB, BEO_1, with value 0x5
  unsigned long HxSS; // DAQ header, 4 LSB, Hx$$, with value 0x8 (or 0x0)
  unsigned long Resv; // TK header,  8 MSB, with value 0xED (???)
  
  if ( headerBytes_ < 0 ) { // Use "search mode" to find DAQ header
    
    for ( uint16_t ichar = 0; ichar < input.size()-16; ichar++ ) { 
      unsigned long* input_u32 = reinterpret_cast<unsigned long*>( const_cast<unsigned char*>( input.data() ) + ichar );
      BOE1 = input_u32[0] & 0xF0000000;
      HxSS = input_u32[1] & 0x0000000F;
      Resv = input_u32[2] & 0xFF000000;
      if ( BOE1 == 0x50000000 &&
	   HxSS == 0x00000008 ) { // && Resv == 0xED000000 ) {
	cout << "[SiStripRawToDigi::locateStartOfFedBuffer]" 
		  << " FED buffer has been found at byte position " 
	     << ichar << " with a size of " 
	     << input.size()-ichar << " bytes" << endl;
	cout << "[SiStripRawToDigi::locateStartOfFedBuffer]" 
	     << " adjust the configurable 'AppendedHeaderBytes' to " 
	     << ichar << endl;
	// Found DAQ header at byte position 'ichar' 
	// Return adjusted buffer start position and size
	output.resize( input.size()-ichar );
	memcpy( output.data(),        // target
		input.data()+ichar,   // source
		input.size()-ichar ); // nbytes
      }
    }
    // Didn't find DAQ header after search
    // Return UNadjusted buffer start position and size
    output.resize( input.size() );
    memcpy( output.data(), input.data(), input.size() );

  } else { 

    // Adjust according to the 'AppendedHeaderBytes' configurable
    unsigned long* input_u32 = reinterpret_cast<unsigned long*>( const_cast<unsigned char*>( input.data() ) + headerBytes_ );
    BOE1 = input_u32[0] & 0xF0000000;
    HxSS = input_u32[1] & 0x0000000F;
    Resv = input_u32[2] & 0xFF000000;
    if ( !( BOE1 == 0x50000000 &&
	    HxSS == 0x00000008 ) ) { 
      stringstream ss;
      ss << "[SiStripRawToDigi::locateStartOfFedBuffer]"
	 << " DAQ header not found at expected location!"
	 << " First 64-bit word of buffer is 0x"
	 << hex 
	 << setfill('0') << setw(8) << input_u32[0] 
	 << setfill('0') << setw(8) << input_u32[1] 
	 << dec
	 << ". Adjust 'AppendedHeaderBytes' configurable"
	 << " to negative value to activate 'search mode'" << endl;
      throw string( ss.str() ); 
      // DAQ header not found at expected location
      // Return UNadjusted buffer start position and size
      output.resize( output.size() );
      memcpy( output.data(), input.data(), input.size() );
    } else {
      // DAQ header found at expected location
      // Return adjusted buffer start position and size
      output.resize( input.size()-headerBytes_ );
      memcpy( output.data(), 
	      input.data()+headerBytes_, 
	      input.size()-headerBytes_ );
    }
  }
  
  // Check on size of output buffer
  if ( output.size() < 24 ) { 
    stringstream ss; 
    ss << "[SiStripRawToDigi::locateStartOfFedBuffer] "
       << "Output FEDRawData with FED id " << fed_id 
       << " has size " << output.size() << endl;
    throw string( ss.str() );
    throw string("[SiStripRawToDigi::locateStartOfFedBuffer] should never get here!"); 
  } 
  
}

// -----------------------------------------------------------------------------
/** 
    Iterate through DetIds within collection and their digis,
    increment various counters and "histogram" positions and ADC
    values of digis.
*/
void SiStripRawToDigi::recordDebugInfo( auto_ptr< edm::DetSetVector<SiStripRawDigi> >& scope_mode,
					auto_ptr< edm::DetSetVector<SiStripRawDigi> >& virgin_raw,
					auto_ptr< edm::DetSetVector<SiStripRawDigi> >& proc_raw,
					auto_ptr< edm::DetSetVector<SiStripDigi> >& zero_suppr ) {
  cout << "[SiStripRawToDigi::recordDebugInfo]" << endl;
//   vector<unsigned int> dets; digis->detIDs( dets );
//   unsigned long ndigis = 0; // Digi counter within event
//   unsigned long ndets = 0; // Dets counter within event
//   vector<unsigned int>::iterator idet;
//   for( idet = dets.begin(); idet != dets.end(); idet++ ) {
//     if ( !(*idet) ) { 
//       throw string("[SiStripRawToDigi::recordDebugInfo] Null DetId in collection!");
//       continue; 
//     }
//     ndets++; nDets_++; 
//     vector<StripDigi> temp; digis->digis(*idet, temp);
//     vector<StripDigi>::const_iterator idigi;
//     for ( idigi = temp.begin(); idigi != temp.end(); idigi++ ) {
//       ndigis++; nDigis_++;
//       unsigned int strip = (*idigi).channel(); // starts from 0
//       if ( strip >= position_.size() ) { position_.resize(strip+1,0); }
//       position_[ strip ]++; 
//       unsigned int value = (*idigi).adc(); // starts from 0
//       if ( value >= landau_.size() ) { landau_.resize(value+1,0); }
//       landau_[ value ]++; 
//     }
//   }
//   if ( ndigis ) {
//     cout << "[SiStripRawToDigi::recordDebugInfo] "
// 	 << "Extracted " << ndigis
// 	 << " digis from " << ndets
// 	 << " detectors " << endl;	
//   }
}

//------------------------------------------------------------------------------
/** 
    Dumps raw data to stdout (NB: payload is byte-swapped,
    headers/trailer are not).
*/
void SiStripRawToDigi::dumpRawData( uint16_t fed_id, 
				    const FEDRawData& buffer ) {
  cout << "[SiStripRawToDigi::dumpRawData] "
       << "Dump of buffer from FED id " <<  fed_id 
       << " which contains " << buffer.size() 
       <<" bytes" << endl;
  cout << "NB: payload is byte-swapped and empty words have been removed." << endl;
  unsigned long* buffer_u32 = reinterpret_cast<unsigned long*>( const_cast<unsigned char*>( buffer.data() ) );
  unsigned int empty = 0;
  for ( unsigned long i = 0; i < buffer.size()/8; i++ ) {
    unsigned int temp0 = buffer_u32[i*2] & 0xFFFFFFFF;
    unsigned int temp1 = buffer_u32[i*2+1] & 0xFFFFFFFF;
    if ( !temp0 && !temp1 ) { empty++; }
    else { 
      if ( empty ) { 
	cout << "       [ empty  words ]" << endl; 
	empty = 0; 
      }
      cout << setw(5) << dec << i*8 << ": " << hex 
	   << setfill('0') << setw(8) << temp0 
	   << setfill('0') << setw(8) << temp1 
	   << dec << endl;
    }
  }
  cout << "[SiStripRawToDigi::dumpRawData] "
       << "End of FED buffer" << endl << endl;
}


