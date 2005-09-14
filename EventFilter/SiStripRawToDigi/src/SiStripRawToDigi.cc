#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToDigi.h"
//
#include <iostream>
#include<vector>

// -----------------------------------------------------------------------------
// constructor
SiStripRawToDigi::SiStripRawToDigi( SiStripConnection& connections,
				    unsigned short verbosity ) : 
  connections_(), // cabling map
  verbosity_(verbosity), 
  description_(0), fedEvent_(0), // Fed9U classes
  readoutPath_("SLINK"), readoutMode_("ZERO_SUPPRESSED"),
  fedids_(), // FED identifier list
  position_(), landau_(), // debug counters
  nFeds_(0), nDets_(0), nDigis_(0) // debug counters
{
  if (verbosity_>1) std::cout << "[SiStripRawToDigi::SiStripRawToDigi] " 
			      << "constructing SiStripRawToDigi object..." << std::endl;

  // initialisation of cabling map object
  connections_ = connections;

  // initialise container holding FED ids.
  fedids_.clear(); fedids_.reserve(500);

  // initialise some containers holding debug info
  landau_.clear(); landau_.reserve(100); landau_.resize(100,0);
  position_.clear(); position_.reserve(512); position_.resize(512,0);
  
  // extract list of FED ids 
  vector<unsigned short> feds; // temp container
  connections_.getConnectedFedNumbers( feds ); //@@ arg should read "fedids_"
  
  //@@ below is temporary due to bug in SiStripConnections class!
  std::vector<unsigned short>::iterator iter; 
  for ( iter = feds.begin(); iter != feds.end(); iter++) {
    bool new_id = true;
    std::vector<unsigned short>::iterator ifed;
    for ( ifed = fedids_.begin(); ifed != fedids_.end(); ifed++ ) {
      if (*ifed == *iter) { new_id = false; break; }
    }
    if ( new_id ) { fedids_.push_back(*iter); }
  }
  //some debug
  if (verbosity_>2) { 
    std::cout << "[SiStripRawToDigi::createDigis] "
	      << "Number of FED ids: " << fedids_.size() << ", "
	      << "List of FED ids: ";
    for ( unsigned int ifed = 0; ifed < fedids_.size(); ifed++ ) { 
      std::cout << fedids_[ifed] << ", ";
    }
    cout << std::endl;
  }
  
  //some debug
  if (verbosity_>2) { 
    std::map< unsigned short, vector<cms::DetId> > partitions;
    std::map< unsigned short, std::vector<cms::DetId> >::iterator ifed;
    connections_.getDetPartitions( partitions );
    std::cout << "[SiStripRawToDigi::SiStripDigiToRaw] "
	      << "Number of FED \"partitions\": " 
	      << partitions.size() << std::endl
	      << " FedId/nDets: ";
    for ( ifed = partitions.begin(); ifed != partitions.end(); ifed++ ) { 
      std::cout << (*ifed).first << "/"
		<< ((*ifed).second).size() << ", ";
    }
    cout << std::endl;
  }
  
}

// -----------------------------------------------------------------------------
// destructor
SiStripRawToDigi::~SiStripRawToDigi() {
  if (verbosity_>1) std::cout << "[SiStripRawToDigi::~SiStripRawToDigi] " 
			      << "destructing SiStripRawToDigi object..." << std::endl;

  // counters
  std::cout << "[SiStripRawToDigi::~SiStripRawToDigi] Some cumulative counters: "
	    << "#FEDs: " << nFeds_ 
	    << "  #Dets: " << nDets_ 
	    << "  #Digis_: " << nDigis_ << std::endl;

  // ndigis
  std::cout << "[SiStripRawToDigi::~SiStripRawToDigi] "
	    << "Digi statistics (vs strip position): " << std::endl;
  int tmp1 = 0;
  for (unsigned int i=0; i<position_.size(); i++) {
    if ( i<10 ) { 
      std::cout << "Strip: " << i << ",  Digis: " << position_[i] << std::endl; 
    }
    tmp1 += position_[i];
  }
  std::cout << "Ndigis: " << tmp1 << std::endl;

  // landau
  std::cout << "[SiStripRawToDigi::~SiStripRawToDigi] Landau statistics: " << std::endl;
  int tmp2 = 0;
  for (unsigned int i=0; i<landau_.size(); i++) {
    if ( i<10 ) { 
      std::cout << "ADC: " << i << ",  Digis: " << landau_[i] << std::endl; 
    }
    tmp2 += landau_[i];
  }
  std::cout << "Ndigis: " << tmp1 << std::endl;
  
}

// -----------------------------------------------------------------------------
// method to create a FEDRawDataCollection using a StripDigiCollection as input
void SiStripRawToDigi::createDigis( raw::FEDRawDataCollection& fed_buffers,
				    StripDigiCollection& digis ) { 
  if (verbosity_>2) std::cout << "[SiStripRawToDigi::createDigis] " << std::endl;

  try {
    
    // some temporary debug...
    if (verbosity_>2) {
      vector<unsigned int> dets = digis.detIDs();
      int filled = 0;
      for ( int ifed = 0; ifed < 1023; ifed++ ) {
	if ( ( fed_buffers.FEDData(ifed) ).data_.size() ) { filled++; } 
      }
      std::cout << "[SiStripRawToDigi::createDigis] " 
		<< "Number of FEDRawData objects is " << filled << ", "
		<< "Number of detectors with digis: " 
		<< dets.size() << std::endl;
    }
    
    // loop through FEDs and extract payload
    std::vector<unsigned short>::iterator ifed;
    for ( ifed = fedids_.begin(); ifed != fedids_.end(); ifed++ ) {

      if (verbosity_>2) { 
	std::cout << "[SiStripRawToDigi::createDigis] "
		  << "Extracting digis from FED id: " << *ifed << std::endl;
      }
      
      // extract FEDRawData structure from FEDRawDataCollection
      raw::FEDRawData& fed_buffer = fed_buffers.FEDData( static_cast<int>(*ifed) );
      
      //// get the data buffer (in I8 format), reinterpret as array in U32 format (as
      //// required by Fed9UEvent) and get buffer size (convert from units of I8 to U32)
      unsigned char* buffer = const_cast<unsigned char*>(fed_buffer.data());
      unsigned long* buffer_u32 = reinterpret_cast<unsigned long*>( buffer );
      unsigned long size = (static_cast<unsigned long>(fed_buffer.data_.size())) / 4; 

      // dump FED buffer to stdout
      if (verbosity_>2) { dumpFedBuffer( *ifed, buffer, size ); }

      // check if FED buffer is of non-zero size
      if ( !size ) { 
	
	stringstream os; 
	os << "[SiStripRawToDigi::createDigis] "
	   << "Buffer of FED id " << *ifed << " is of zero size";
	throw string( os.str() );
	continue; 
      } 

      // counter for debug purposes
      nFeds_++; 
      
      if (verbosity_>2) { 
	std::cout << "[SiStripRawToDigi::createDigis] "
		  << "FED buffer size (32 bit words): " << size << std::endl;
      }
      
      // remove VME header (if present) from start of FED buffer
      if ( readoutPath_ == "VME" ) { 
	unsigned int shift = buffer_u32[11];
	unsigned int nchan = buffer_u32[13];
	unsigned int start = 10 + shift + nchan;
	size -= start; // recalculate buffer size (size after "start" pos)
      }   
      
      // create new Fed9UEvent object using present FED buffer
      fedEvent_ = new Fed9U::Fed9UEvent(); //@@ temporary
      fedEvent_->Init(buffer_u32, description_, size); 
      
      // get FED readout mode from the FED buffer (using Fed9UEvent)
      // and call appropriate method to extract payload

      //readoutMode_ = static_cast<unsigned short>( fedEvent_->getSpecialTrackerEventType() );
      //readoutMode_ = fedEvent_->getDaqMode();
      if (verbosity_>2) { std::cout << "Readout mode : " << readoutMode_ << std::endl; }
      if ( readoutMode_ == "SCOPE_MODE" )           { scopeMode( *ifed, digis ); }
      else if ( readoutMode_ == "VIRGIN_RAW" )      { virginRaw( *ifed, digis ); } 
      else if ( readoutMode_ == "PROCESSED_RAW" )   { procRaw( *ifed, digis ); }
      else if ( readoutMode_ == "ZERO_SUPPRESSED" ) { zeroSuppr( *ifed, digis ); } 
      else { std::cout << "Unknown readout mode : " << readoutMode_ << std::endl; }

//       //readoutMode_ = static_cast<unsigned short>( fedEvent_->getSpecialTrackerEventType() );
//       //readoutMode_ = fedEvent_->getDaqMode();
//       std::cout << "Readout mode : " << readoutMode_ << std::endl;
//       if ( readoutMode_ == 1/*3*/ )       { scopeMode( *ifed, digis ); }
//       else if ( readoutMode_ == 2/*2*/ )  { virginRaw( *ifed, digis ); } 
//       else if ( readoutMode_ == 6/*0*/ )  {   procRaw( *ifed, digis ); }
//       else if ( readoutMode_ == 10/*1*/ ) { zeroSuppr( *ifed, digis ); } 
//       else { std::cout << "Unknown readout mode : " << readoutMode_ << std::endl; }

    } // loop over FED ids

    // some debug
    if (verbosity_>0) { 
      vector<unsigned int> dets = digis.detIDs();
      long cntr = 0;
      std::vector<unsigned int>::iterator idet;
      for( idet = dets.begin(); idet != dets.end(); idet++ ) {
	nDets_++;
	StripDigiCollection::Range temp = digis.get(*idet);
	StripDigiCollection::ContainerIterator idigi;
	for ( idigi = temp.first; idigi != temp.second; idigi++ ) {
	  if ( (*idigi).adc() ) {
	    position_[ (*idigi).channel() ]++; 
	    landau_[ (*idigi).adc()<100 ? (*idigi).adc() : 0 ]++;
	    cntr++;
	    nDigis_++;
	  }
	}
      }
      if ( cntr ) {
// 	std::cout << "[SiStripRawToDigi::createDigis] "
// 		  << "Extracting " << cntr
// 		  << " digis from FED id " << *ifed << std::endl;	
      }
    }
    // end of debug
    
  }
  catch ( string err ){
    std::cout << "[SiStripRawToDigi::createDigis] "
	      << "Exception caught : " << err << std::endl;
  }
  
  delete fedEvent_; //@@ temporary
  
}

// -----------------------------------------------------------------------------
//
void SiStripRawToDigi::zeroSuppr( unsigned short fed_id, StripDigiCollection& digis ) {
  if (verbosity_>2) std::cout << "[SiStripRawToDigi::zeroSuppr] " << std::endl;

  if (1) { // Loop through DetIds for given FED...
    
    // Retrieve detector ids associated with given FED 
    std::vector<cms::DetId> dets;
    connections_.getDetIds( fed_id, 96, dets );
    if ( dets.empty() ) { /* need warning here */ }
    
    // Loop through detector ids 
    std::vector<cms::DetId>::iterator idet;
    for ( idet = dets.begin(); idet != dets.end(); idet++ ) {
      
      // Retrieve Digis from given DetId
      unsigned int det_id = static_cast<unsigned int>( (*idet).rawId() );
      if ( !det_id ) { continue; }
      
      vector<unsigned short> fed_channels;
      connections_.getFedIdAndChannels( det_id, fed_channels );
      if ( fed_channels.empty() ) { continue; }
      
      // StripDigi container for extracted ADC values
      std::vector<StripDigi> channel_digis;
      channel_digis.clear();
      channel_digis.reserve( 256*fed_channels.size() );
      
      // Loop through FED channels of given DetId
      unsigned short apv_pair = 0;
      vector<unsigned short>::iterator ichan;
      for ( ichan = fed_channels.begin(); ichan != fed_channels.end(); ichan++ ) {
	// Iterate through payload for given channel
	
	Fed9U::Fed9UEventIterator fed_iter = const_cast<Fed9U::Fed9UEventChannel&>(fedEvent_->channel( *ichan )).getIterator();
	for (Fed9U::Fed9UEventIterator i = fed_iter+7; i.size() > 0; /**/) {
	  unsigned char first_strip = *i++; // first strip position of cluster
	  unsigned char width = *i++; // strip width of cluster 
	  for (unsigned short istr = 0; istr < width; istr++) {
	    unsigned short strip = apv_pair*256 + first_strip + istr;
	    channel_digis.push_back( StripDigi(static_cast<int>(strip),static_cast<int>(*i)) );
	    *i++;
	  }
	}
	apv_pair++;

      }
      
      // Write StripDigi's to container
      StripDigiCollection::ContainerIterator begin = channel_digis.begin();
      StripDigiCollection::ContainerIterator end = channel_digis.end();
      StripDigiCollection::Range digi_ptrs( begin, end );
      if ( !channel_digis.empty() ) { digis.put( digi_ptrs, det_id ); }
      
    }
    
  } else { // ...or loop through FED channels for given FED

    // loop through FED channels
    for ( unsigned short ichan = 0; ichan < fedEvent_->totalChannels(); ichan++ ) {
      
      // retrieve DetId and APV pair from SiStripConnections
      pair<cms::DetId,unsigned short> det_pair;
      connections_.getDetPair( fed_id, ichan, det_pair );
      unsigned int det_id = static_cast<unsigned int>( det_pair.first.rawId() );
      unsigned short apv_pair = det_pair.second;
      
      // StripDigi container for extracted ADC values
      std::vector<StripDigi> channel_digis;
      channel_digis.clear();
      channel_digis.reserve(256);
      
      // Iterate through payload for given channel
      Fed9U::Fed9UEventIterator fed_iter = const_cast<Fed9U::Fed9UEventChannel&>(fedEvent_->channel( ichan )).getIterator();
      for (Fed9U::Fed9UEventIterator i = fed_iter+7; i.size() > 0; /**/) {
	unsigned char first_strip = *i++; // first strip position of cluster
	unsigned char width = *i++; // strip width of cluster 
	for (unsigned short istr = 0; istr < width; istr++) {
	  unsigned short strip = apv_pair*256 + first_strip + istr;
	  channel_digis.push_back( StripDigi(static_cast<int>(strip),static_cast<int>(*i)) );
	  *i++;
	}
      }

      // Write StripDigi's to container
      StripDigiCollection::ContainerIterator begin = channel_digis.begin();
      StripDigiCollection::ContainerIterator end = channel_digis.end();
      StripDigiCollection::Range digi_ptrs( begin, end );
      if ( !channel_digis.empty() ) { digis.put( digi_ptrs, det_id ); }

    }
    
  } 
  
}

// -----------------------------------------------------------------------------
//
void SiStripRawToDigi::virginRaw( unsigned short fed_id, StripDigiCollection& digis ) {

  if (verbosity_>2) std::cout << "[SiStripRawToDigi::virginRaw] " << std::endl;

  if (1) { // Loop through DetIds for given FED...
    
    // Retrieve detector ids associated with given FED 
    std::vector<cms::DetId> dets;
    connections_.getDetIds( fed_id, 96, dets );
    if ( dets.empty() ) { /* need warning here */ }
    
    // Loop through detector ids 
    std::vector<cms::DetId>::iterator idet;
    for ( idet = dets.begin(); idet != dets.end(); idet++ ) {
      
      // Retrieve Digis from given DetId
      unsigned int det_id = static_cast<unsigned int>( (*idet).rawId() );
      if ( !det_id ) { continue; }
      
      vector<unsigned short> fed_channels;
      connections_.getFedIdAndChannels( det_id, fed_channels );
      if ( fed_channels.empty() ) { continue; }
      
      // StripDigi container for extracted ADC values
      std::vector<StripDigi> channel_digis;
      channel_digis.clear();
      channel_digis.reserve( 256*fed_channels.size() );
      
      // Loop through FED channels of given DetId
      unsigned short apv_pair = 0;
      vector<unsigned short>::iterator ichan;
      for ( ichan = fed_channels.begin(); ichan != fed_channels.end(); ichan++ ) {

	// Retrieve data samples from FED channel
	vector<unsigned short> adc = fedEvent_->channel( *ichan ).getSamples();
	if ( adc.size() != 256 ) { 
	  stringstream os;
	  os << "[SiStripRawToDigi::virginRaw] "
	     << "Warning : Number of ADC samples from FED channel != 256";
	  throw string( os.str() );
	}
	
	// Account for APV readout order and multiplexed data
	for ( unsigned short i = 0; i < adc.size(); i++ ) {
	  int j = readoutOrder(i%128);
	  (i/128) ? j=j*2+1 : j=j*2; // true=APV1, false=APV0
	  unsigned short strip = apv_pair*256 + i;
	  channel_digis.push_back( StripDigi(strip, adc[j]) );
	}

	apv_pair++;
      }

      // Write StripDigi's to container
      StripDigiCollection::ContainerIterator begin = channel_digis.begin();
      StripDigiCollection::ContainerIterator end = channel_digis.end();
      StripDigiCollection::Range digi_ptrs( begin, end );
      if ( !channel_digis.empty() ) { digis.put( digi_ptrs, det_id ); }

    }

  } else { // ...or loop through FED channels for given FED

    // Loop through FED channels
    for ( unsigned short ichan = 0; ichan < fedEvent_->totalChannels(); ichan++ ) {
      
      // Retrieve DetId and APV pair from SiStripConnections
      pair<cms::DetId,unsigned short> det_pair;
      connections_.getDetPair( fed_id, ichan, det_pair );
      unsigned int det_id = static_cast<unsigned int>( det_pair.first.rawId() );
      unsigned short apv_pair = det_pair.second;

      // Retrieve data samples from FED channel
      vector<unsigned short> adc = fedEvent_->channel( ichan ).getSamples();
      if ( adc.size() != 256 ) { 
	stringstream os;
	os << "[SiStripRawToDigi::virginRaw] "
	   << "Warning : Number of ADC samples from FED channel != 256";
	throw string( os.str() );
      }
      
      // StripDigi container for extracted ADC values
      vector<StripDigi> channel_digis; 
      channel_digis.clear(); 
      channel_digis.reserve( 256 );

      // Account for APV readout order and multiplexed data
      for ( unsigned short i = 0; i < adc.size(); i++ ) {
	int j = readoutOrder(i%128);
	(i/128) ? j=j*2+1 : j=j*2; // true=APV1, false=APV0
	unsigned short strip = apv_pair*256 + j;
	channel_digis.push_back( StripDigi(strip, adc[j]) );
      }

      // Write StripDigi's to container
      StripDigiCollection::ContainerIterator begin = channel_digis.begin();
      StripDigiCollection::ContainerIterator end = channel_digis.end();
      StripDigiCollection::Range digi_ptrs( begin, end );
      if ( !channel_digis.empty() ) { digis.put( digi_ptrs, det_id ); }
    }
  
  }

}

// -----------------------------------------------------------------------------
//
void SiStripRawToDigi::scopeMode( unsigned short fed_id, StripDigiCollection& digis ) {
  /* needs implementation */ 
  stringstream os;
  os << "[SiStripRawToDigi::scopeMode] "
     << " This method has not been implemented yet!";
  throw string( os.str() );
}

// -----------------------------------------------------------------------------
//
void SiStripRawToDigi::procRaw( unsigned short fed_id, StripDigiCollection& digis ) {
  /* needs implementation */ 
  stringstream os;
  os << "[SiStripRawToDigi::procRaw] "
     << " This method has not been implemented yet!";
  throw string( os.str() );
}

//------------------------------------------------------------------------------
// DEBUG: dump of FED buffer to stdout (NB: payload is byte-swapped, headers/trailer are not)
void SiStripRawToDigi::dumpFedBuffer( unsigned short fed, unsigned char* start, unsigned long size ) {
  std::cout << "[SiStripRawToDigi::dumpFedBuffer] "
	    << "Dump of FED buffer to stdout:" << std::endl;
  std::cout << "Reinterpreting FED data as " 
	    << 8*sizeof(unsigned long long int) << "-bit words" << std::endl;
  std::cout << "Buffer from FED id: " << dec << fed 
	    << " has length [64-bit words]: " << dec << size/8 << std::endl;
  std::cout << "NB: payload is byte-swapped and empty words have been removed." << std::endl;
  unsigned int* buffer_u32 = reinterpret_cast<unsigned int*>( start );
  unsigned int empty = 0;
  for ( unsigned long i = 0; i < size/8; i++ ) {
    unsigned int temp0 = buffer_u32[i*2] & 0xFFFFFFFF;
    unsigned int temp1 = buffer_u32[i*2+1] & 0xFFFFFFFF;
    if ( !temp0 && !temp1 ) { empty++; }
    else { 
      if ( empty ) { 
	std::cout << "     : [ empty  words ]" << std::endl; 
	empty = 0; 
      }
      std::cout << setw(5) << dec << i << ": " << hex 
		<< setfill('0') << setw(8) << temp0 
		<< setfill('0') << setw(8) << temp1 
		<< dec << std::endl;
    }
  }
  std::cout << "[SiStripRawToDigi::dumpFedBuffer] "
	    << "End of FED buffer" << std::endl << std::endl;
}
