#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToDigi.h"
#include "Utilities/Timing/interface/TimingReport.h"
#include "CondFormats/SiStripObjects/interface/SiStripReadoutCabling.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiStripDigi/interface/StripDigiCollection.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>

// -----------------------------------------------------------------------------
// constructor
SiStripRawToDigi::SiStripRawToDigi( unsigned short verbosity ) : 
  verbosity_(verbosity), 
  fedDescription_(0), fedEvent_(0),
  readoutPath_("SLINK"), readoutMode_("ZERO_SUPPRESSED"),
  position_(), landau_(), // debug counters
  nFeds_(0), nDets_(0), nDigis_(0) // debug counters
{
  if ( verbosity_>0 ) {
    std::cout << "[SiStripRawToDigi::SiStripRawToDigi] " 
	      << "Constructing object..." << std::endl;
  }
  // initialise some containers holding debug info
  landau_.clear(); landau_.reserve(100); landau_.resize(100,0);
  position_.clear(); position_.reserve(512); position_.resize(512,0);
  

//   //some debug
//    if (verbosity_>2) { 
//      std::cout << "[SiStripRawToDigi::createDigis] "
//  	      << "Number of FED ids: " << fedids_.size() << ", "
//  	      << "List of FED ids: ";
//      for ( unsigned int ifed = 0; ifed < fedids_.size(); ifed++ ) { 
//        std::cout << fedids_[ifed] << ", ";
//      }
//      cout << std::endl;
//    }
  
//   //some debug
//    if (verbosity_>2) { 
//      std::map< unsigned short, vector<DetId> > partitions;
//      std::map< unsigned short, std::vector<DetId> >::iterator ifed;
//      connections_.getDetPartitions( partitions );
//      std::cout << "[SiStripRawToDigi::SiStripDigiToRaw] "
//  	      << "Number of FED \"partitions\": " 
//  	      << partitions.size() << std::endl
//  	      << " FedId/nDets: ";
//      for ( ifed = partitions.begin(); ifed != partitions.end(); ifed++ ) { 
//        std::cout << (*ifed).first << "/"
//  		<< ((*ifed).second).size() << ", ";
//      }
//      cout << std::endl;
//    }
  
}

// -----------------------------------------------------------------------------
// destructor
SiStripRawToDigi::~SiStripRawToDigi() {
  if ( verbosity_>0 ) {
    std::cout << "[SiStripRawToDigi::~SiStripRawToDigi]" 
	      << " Destructing object..." << std::endl;
  }
  // counters
  if ( verbosity_>0 ) {
    std::cout << "[SiStripRawToDigi::~SiStripRawToDigi] Some cumulative counters: "
	      << "#FEDs: " << nFeds_ 
	      << "  #Dets: " << nDets_ 
	      << "  #Digis_: " << nDigis_ << std::endl;
  }
  if ( verbosity_>2 ) {
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
}

// -----------------------------------------------------------------------------
// method to create a FEDRawDataCollection using a StripDigiCollection as input
void SiStripRawToDigi::createDigis( edm::ESHandle<SiStripReadoutCabling>& cabling,
				    edm::Handle<FEDRawDataCollection>& buffers,
				    std::auto_ptr<StripDigiCollection>& digis ) {

  if ( verbosity_>1 ) {
    std::cout << "[SiStripRawToDigi::createDigis]" << std::endl;
  }

  try {

    
    // Some debug
    if (verbosity_>2) {
      int filled = 0;
      for ( int ifed = 0; ifed < 1023; ifed++ ) { 
 	if ( ( buffers->FEDData(ifed) ).size() ) { filled++; } 
      }
      std::vector<unsigned int> det_ids;
      digis->detIDs( det_ids );
      std::cout << "[SiStripRawToDigi::createDigis] " 
 		<< "Number of FEDRawData objects is " << filled << ", "
 		<< "Number of detectors with digis: " 
 		<< det_ids.size() << std::endl;
    }
    
    // Retrieve FED ids, iterate through FEDs and extract payload
    const std::vector<unsigned short> fed_ids = cabling->getFEDs();
    std::vector<unsigned short>::const_iterator ifed;
    for ( ifed = fed_ids.begin(); ifed != fed_ids.end(); ifed++ ) {

      if ( verbosity_>1 ) { 
 	std::cout << "[SiStripRawToDigi::createDigis] "
 		  << "Extracting digis from FED id: " << *ifed << std::endl;
      }

      // Extract FEDRawData from collection
      const FEDRawData& fed_buffer = buffers->FEDData( static_cast<int>(*ifed) );
     
      // Get the data buffer (in I8 format), reinterpret as array in U32 format (as
      // required by Fed9UEvent) and get buffer size (convert from units of I8 to U32)
      unsigned char* buffer = const_cast<unsigned char*>(fed_buffer.data());
      unsigned long* buffer_u32 = reinterpret_cast<unsigned long*>( buffer );
      unsigned long size = (static_cast<unsigned long>(fed_buffer.size())) / 4; 

      // dump FED buffer to stdout
      if (verbosity_>3) { dumpFedBuffer( *ifed, buffer, size*4 ); }

      // check if FED buffer is of non-zero size
      if ( !size ) { 
	std::stringstream os; 
 	os << "[SiStripRawToDigi::createDigis] "
 	   << "Buffer of FED id " << *ifed << " is of zero size";
 	throw string( os.str() );
 	continue; 
      } 
      
      // counter for debug purposes
      nFeds_++; 
      
      if ( verbosity_>1 ) { 
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
      fedEvent_->Init(buffer_u32, fedDescription_, size); 
   
      // get FED readout mode from the FED buffer (using Fed9UEvent)
      // and call appropriate method to extract payload
 
      //readoutMode_ = static_cast<unsigned short>( fedEvent_->getSpecialTrackerEventType() );
      //readoutMode_ = fedEvent_->getDaqMode();
      if (verbosity_>2) { std::cout << "Readout mode : " << readoutMode_ << std::endl; }
      if ( readoutMode_ == "SCOPE_MODE" )           { scopeMode( *ifed, cabling, digis ); }
      else if ( readoutMode_ == "VIRGIN_RAW" )      { rawModes( *ifed, cabling, digis ); } 
      else if ( readoutMode_ == "PROCESSED_RAW" )   { rawModes( *ifed, cabling, digis ); }
      else if ( readoutMode_ == "ZERO_SUPPRESSED" ) { zeroSuppr( *ifed, cabling, digis ); } 
      else { std::cout << "Unknown readout mode : " << readoutMode_ << std::endl; }
      
      //readoutMode_ = static_cast<unsigned short>( fedEvent_->getSpecialTrackerEventType() );
      //readoutMode_ = fedEvent_->getDaqMode();
      //std::cout << "Readout mode : " << readoutMode_ << std::endl;
      //if ( readoutMode_ == 1/*3*/ )       { scopeMode( *ifed, digis ); }
      //else if ( readoutMode_ == 2/*2*/ )  { virginRaw( *ifed, digis ); } 
      //else if ( readoutMode_ == 6/*0*/ )  {   procRaw( *ifed, digis ); }
      //else if ( readoutMode_ == 10/*1*/ ) { zeroSuppr( *ifed, digis ); } 
      //else { std::cout << "Unknown readout mode : " << readoutMode_ << std::endl; }

      if ( fedEvent_ ) { delete fedEvent_; }

    } // loop over FED ids
   
    //  some debug
     if (verbosity_>0) { 
       std::vector<unsigned int> dets;
       digis->detIDs(dets);
       long cntr = 0;
       std::vector<unsigned int>::iterator idet;
       for( idet = dets.begin(); idet != dets.end(); idet++ ) {
 	nDets_++;
 	std::vector<StripDigi> temp;
	digis->digis(*idet, temp);
 	std::vector<StripDigi>::const_iterator idigi;
 	for ( idigi = temp.begin(); idigi != temp.end(); idigi++ ) {
 	  if ( (*idigi).adc() ) {
 	    position_[ (*idigi).channel() ]++; 
 	    landau_[ (*idigi).adc()<100 ? (*idigi).adc() : 0 ]++;
 	    cntr++;
 	    nDigis_++;
 	  }
 	}
       }
       if ( cntr ) {
 	std::cout << "[SiStripRawToDigi::createDigis] "
  		  << "Extracting " << cntr
  		  << " digis from event" << std::endl;	
       }
     }
    // end of debug
    
  }
  catch ( string err ){
    std::cout << "[SiStripRawToDigi::createDigis] "
	      << "Exception caught : " << err << std::endl;
  }
  
}

// -----------------------------------------------------------------------------
//
void SiStripRawToDigi::zeroSuppr( unsigned short fed_id, 
				  edm::ESHandle<SiStripReadoutCabling>& cabling,
				  std::auto_ptr<StripDigiCollection>& digis ) {

  if (verbosity_>2) std::cout << "[SiStripRawToDigi::zeroSuppr] " << std::endl;

  // Loop through all channels of given FED id
  for ( unsigned short ichan = 0; ichan < fedEvent_->totalChannels(); ichan++ ) {
    
    // Retrieve pair containing DetId and APV pair number
    pair<unsigned int, unsigned short> apv_pair_id = cabling->getAPVPair( fed_id, ichan );

    // Check DetId is non-zero
    if ( !apv_pair_id.first ) { 
      if ( verbosity_ > 2 ) { 
	std::cout  << "[SiStripRawToDigi::zeroSuppr]"
		   << " Zero DetId returned for FED id " << fed_id
		   << " and channel " << ichan << std::endl; 
      }
      continue;
    }
      
    // Define digi container
    std::vector<StripDigi> channel_digis;
    channel_digis.clear(); channel_digis.reserve(256);
    
    // Retrieve channel payload and create digis
    Fed9U::Fed9UEventIterator fed_iter = const_cast<Fed9U::Fed9UEventChannel&>(fedEvent_->channel( ichan )).getIterator();
    for (Fed9U::Fed9UEventIterator i = fed_iter+7; i.size() > 0; /**/) {
      unsigned char first_strip = *i++; // first strip position of cluster
      unsigned char width = *i++; // strip width of cluster 
      for (unsigned short istr = 0; istr < width; istr++) {
	unsigned short strip = apv_pair_id.second*256 + first_strip + istr;
	channel_digis.push_back( StripDigi(static_cast<int>(strip),static_cast<int>(*i)) );
	*i++;
      }
    }
    digis->add( apv_pair_id.first, channel_digis );
    
    
  }
}

// -----------------------------------------------------------------------------
//
void SiStripRawToDigi::rawModes( unsigned short fed_id, 
				 edm::ESHandle<SiStripReadoutCabling>& cabling,
				 std::auto_ptr<StripDigiCollection>& digis ) {
  
  if (verbosity_>2) { std::cout << "[SiStripRawToDigi::rawModes] " << std::endl; }
  
  // Loop through FED channels of given FED
  for ( unsigned short ichan = 0; ichan < fedEvent_->totalChannels(); ichan++ ) {
    
    // Retrieve pair containing DetId and APV pair number
    pair<unsigned int, unsigned short> apv_pair_id = cabling->getAPVPair( fed_id, ichan );

    // Check DetId is non-zero
    if ( !apv_pair_id.first ) { 
      if ( verbosity_ > 2 ) { 
	std::cout  << "[SiStripRawToDigi::rawModes]"
		   << " Zero DetId returned for FED id " << fed_id
		   << " and channel " << ichan << std::endl; 
      }
      continue;
    }

    // Define digi container
    std::vector<StripDigi> channel_digis;
    channel_digis.clear(); channel_digis.reserve(256);

    // Retrieve channel payload and create digis
    vector<unsigned short> adc = fedEvent_->channel( ichan ).getSamples();
    if ( adc.size() != 256 ) { 
      stringstream os;
      os << "[SiStripRawToDigi::virginRaw] "
	 << "Warning : Number of ADC samples from FED channel != 256";
      throw string( os.str() );
    }
    
    if ( readoutMode_ == "VIRGIN_RAW" ) {
      // Account for APV readout order and multiplexed data
      for ( unsigned short i = 0; i < adc.size(); i++ ) {
	int j = readoutOrder(i%128);
	(i/128) ? j=j*2+1 : j=j*2; // true=APV1, false=APV0
	unsigned short strip = apv_pair_id.second*256 + i;
	channel_digis.push_back( StripDigi(strip, adc[j]) );

      }
    } else if ( readoutMode_ == "PROCESSED_RAW" ) { 
      for ( unsigned short i = 0; i < adc.size(); i++ ) {
	unsigned short strip = apv_pair_id.second*256 + i;
	channel_digis.push_back( StripDigi(strip, adc[i]) );
      }
    } else {
      std::stringstream os; 
      os << "[SiStripRawToDigi::rawModes]"
	 << " Unexpected readout mode!"; 
      throw std::string( os.str() );
    }
    digis->add( apv_pair_id.first, channel_digis );
  
  }  
  
}

// -----------------------------------------------------------------------------
//
void SiStripRawToDigi::scopeMode( unsigned short fed_id, 
				  edm::ESHandle<SiStripReadoutCabling>& cabling,
				  std::auto_ptr<StripDigiCollection>& digis ) {
  /* needs implementation */ 
  std::stringstream os;
  os << "[SiStripRawToDigi::scopeMode] "
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
  std::cout << "Buffer from FED id: " << std::dec << fed 
	    << " has length [64-bit words]: " << std::dec << size/8 << std::endl;
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
      std::cout << std::setw(5) << std::dec << i << ": " << std::hex 
		<< std::setfill('0') << std::setw(8) << temp0 
		<< std::setfill('0') << std::setw(8) << temp1 
		<< std::dec << std::endl;
    }
  }
  std::cout << "[SiStripRawToDigi::dumpFedBuffer] "
	    << "End of FED buffer" << std::endl << std::endl;
}
