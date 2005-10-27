#include "EventFilter/SiStripRawToDigi/interface/SiStripDigiToRaw.h"
#include <iostream>
#include<vector>

// -----------------------------------------------------------------------------
// constructor
SiStripDigiToRaw::SiStripDigiToRaw( SiStripConnection& connections,
				    unsigned short verbosity ) : 
  connections_(),
  verbosity_(verbosity),
  readoutPath_("SLINK"), readoutMode_("VIRGIN_RAW"),
  fedids_(), // FED identifier list
  position_(), landau_(), // debug counters
  nFeds_(0), nDets_(0), nDigis_(0) // debug counters
{
  if (verbosity_>1) std::cout << "[SiStripDigiToRaw::SiStripDigiToRaw] " 
			      << "Constructing object..." << endl;

  // initialisation of cabling map object
  connections_ = connections;

  // initialise container holding FED ids.
  fedids_.clear(); fedids_.reserve( 500 );
  vector<unsigned short> feds; // temp container
  connections_.getConnectedFedNumbers( feds ); //@@ arg should read "fedids_"

  // initialise some containers holding debug info
  landau_.clear(); landau_.reserve(100); landau_.resize(100,0);
  position_.clear(); position_.reserve(512); position_.resize(512,0);
  
  //@@ below is temporary due to bug in SiStripConnections class!
  std::vector<unsigned short>::iterator iter; 
  for ( iter = feds.begin(); iter != feds.end(); iter++) {
    bool new_id = true;
    std::vector<unsigned short>::iterator ifed;
    for ( ifed = fedids_.begin(); ifed != fedids_.end(); ifed++ ) {
      //   if (*ifed == *iter) { new_id = false; break; }
    }
    if ( new_id ) { fedids_.push_back(*iter); }
  }
  //some debug
  if (verbosity_>2) { 
    std::cout << "[SiStripDigiToRaw::createDigis] "
	      << "Number of FED ids: " << fedids_.size() << ", "
	      << "List of FED ids: ";
    for ( unsigned int ifed = 0; ifed < fedids_.size(); ifed++ ) { 
      std::cout << fedids_[ifed] << ", ";
    }
    cout << std::endl;
  }

  //some debug
  if (verbosity_>2) { 
    std::map< unsigned short, std::vector<DetId> > partitions;
    std::map< unsigned short, std::vector<DetId> >::iterator ifed;
    connections_.getDetPartitions( partitions );
    std::cout << "[SiStripDigiToRaw::SiStripDigiToRaw] "
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
SiStripDigiToRaw::~SiStripDigiToRaw() {
  if (verbosity_>1) std::cout << "[SiStripDigiToRaw::~SiStripDigiToRaw] " 
			      << "destructing SiStripDigiToRaw object..." << endl;

  // counters
  std::cout << "[SiStripDigiToRaw::~SiStripDigiToRaw] Some cumulative counters: "
	    << "#FEDs: " << nFeds_ 
	    << "  #Dets: " << nDets_ 
	    << "  #Digis_: " << nDigis_ << std::endl;

  // ndigis

if (verbosity_>0) {
  std::cout << "[SiStripDigiToRaw::~SiStripDigiToRaw] "
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
  std::cout << "[SiStripDigiToRaw::~SiStripDigiToRaw] Landau statistics: " << std::endl;
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
void SiStripDigiToRaw::createFedBuffers( StripDigiCollection& digis,
					 FEDRawDataCollection& fed_buffers ) {
  if (verbosity_>2) std::cout << "[SiStripDigiToRaw::createFedBuffers] " << endl;

  try {
    
    // Some temporary debug...
    if (verbosity_>2) {
      std::vector<unsigned int> dets = digis.detIDs();
      std::cout << "[SiStripDigiToRaw::createFedBuffers] " 
		<< "Number of detectors with digis: " 
		<< dets.size() << std::endl;
    }

    // Define container for (raw) ADC values
    const unsigned short strips_per_fed = 96 * 256; 
    vector<unsigned short> data_buffer; 
    data_buffer.reserve(strips_per_fed);

    // Loop through FEDs and create buffers
    std::vector<unsigned short>::iterator ifed;
    for ( ifed = fedids_.begin(); ifed != fedids_.end(); ifed++ ) {

      // Counter of FEDs for debug purposes
      nFeds_++; 

      // Initialise buffer holding ADC values
      data_buffer.clear();
      data_buffer.resize(strips_per_fed,0);
      
      //loop through FED channels
      for (unsigned short ichan = 0; ichan < 96; ichan++) {

	// retrieve DetId and APV pair from SiStripConnections
      pair<DetId,unsigned short> det_pair;
      connections_.getDetPair( *ifed, ichan, det_pair );
      unsigned int det_id = static_cast<unsigned int>( det_pair.first.rawId() );
      unsigned short apv_pair = det_pair.second;
     
	// Loop through Digis
	StripDigiCollection::Range my_digis = digis.get( det_id );
	StripDigiCollection::ContainerIterator idigi;
	for ( idigi = my_digis.first; idigi != my_digis.second; idigi++ ) {
	  if ((idigi->strip() >= apv_pair*256) && (idigi->strip()< (apv_pair + 1)*256)) {
	    if (idigi->strip() >= (apv_pair + 1)*256) break;

	  // calc strip position (within scope of FED) of digi
	  unsigned short strip = ichan*256 + (*idigi).strip()%256;
	  
	  if ( strip >= strips_per_fed ) {
	    std::cout << "[SiStripDigiToRaw::createFedBuffers] "
		      << "ERROR : strip >= strips_per_fed" << std::endl;
	  }
	  
	  // check if buffer has already been filled with digi ADC value. 
	  // if not or if filled with different value, fill it.
	  if ( data_buffer[strip] ) { // if yes, cross-check values
	    if ( data_buffer[strip] != (*idigi).adc() ) {
	      std::stringstream os; 
	      os << "SiStripDigiToRaw::createFedBuffers(.): " 
		 << "WARNING: Incompatible ADC values in buffer: "
		 << "FED id: " << *ifed << ", FED channel: " << ichan
		 << ", detector strip: " << (*idigi).strip() 
		 << ", FED strip: " << strip
		 << ", ADC value: " << (*idigi).adc()
		 << ", data_buffer["<<strip<<"]: " << data_buffer[strip];
	      std::cout << os.str() << endl;
	    }
	  } else { // if no, update buffer with digi ADC value
	    data_buffer[strip] = (*idigi).adc(); 
	    // debug: update counters
	    if (verbosity_>0) {
	      position_[ (*idigi).strip() ]++;
	      landau_[ (*idigi).adc()<100 ? (*idigi).adc() : 0 ]++;
	      nDigis_++;
	    }
	    if (verbosity_>2) {
	      std::cout << "FED id: " << *ifed << ", FED channel: " << ichan
			<< ", detector strip: " << (*idigi).strip() 
			<< ", FED strip: " << strip << ", ADC value: " << (*idigi).adc() 
			<< ", data_buffer["<<strip<<"]: " << data_buffer[strip] << std::endl;
	    }
	  }
	}
      }
      }
          
      // instantiate appropriate buffer creator object depending on readout mode
      Fed9U::Fed9UBufferCreator* creator = 0;
      if ( readoutMode_ == "SCOPE_MODE" ) {
	throw string("WARNING : Fed9UBufferCreatorScopeMode not implemented yet!");
      } else if ( readoutMode_ == "VIRGIN_RAW" ) {
	creator = new Fed9U::Fed9UBufferCreatorRaw();
      } else if ( readoutMode_ == "PROCESSED_RAW" ) {
	creator = new Fed9U::Fed9UBufferCreatorProcRaw();
      } else if ( readoutMode_ == "ZERO_SUPPRESSED" ) {
	creator = new Fed9U::Fed9UBufferCreatorZS();
      } else {
	std::cout << "WARNING : UNKNOWN readout mode" << endl;
      }
      
      // generate FED buffer and pass to Daq
      Fed9U::Fed9UBufferGenerator generator( creator );
      generator.generateFed9UBuffer( data_buffer );
      
      FEDRawData& fedrawdata = fed_buffers.FEDData( *ifed ); 
      // calculate size of FED buffer in units of bytes (unsigned char)
      int nbytes = generator.getBufferSize() * 4;
      // resize (public) "data_" member of struct FEDRawData
      fedrawdata.resize( nbytes );
      // copy FED buffer to struct FEDRawData using Fed9UBufferGenerator
      unsigned char* chars = const_cast<unsigned char*>( fedrawdata.data() );
      unsigned int* ints = reinterpret_cast<unsigned int*>( chars );
      generator.getBuffer( ints );
      
    }
  }
  catch ( string err ) {
    std::cout << "SiStripDigiToRaw::createFedBuffers] " 
	      << "Exception caught : " << err << std::endl;
  }
  
}

