#include "EventFilter/SiStripRawToDigi/interface/SiStripDigiToRaw.h"
#include "Utilities/Timing/interface/TimingReport.h"
#include "CondFormats/SiStripObjects/interface/SiStripReadoutCabling.h"
#include "DataFormats/SiStripDigi/interface/StripDigiCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include <iostream>
#include <sstream>
#include <vector>

// -----------------------------------------------------------------------------
// constructor
SiStripDigiToRaw::SiStripDigiToRaw( unsigned short verbosity ) : 
  verbosity_(verbosity),
  readoutPath_("SLINK"), readoutMode_("VIRGIN_RAW"),
  position_(), landau_(), // debug counters
  nFeds_(0), nDigis_(0) // debug counters
{
  
  std::cout << "[SiStripDigiToRaw::SiStripDigiToRaw]" 
	    << " Constructing object..." << endl;
  
  // initialise some containers holding debug info
  landau_.clear(); landau_.reserve(100); landau_.resize(100,0);
  position_.clear(); position_.reserve(512); position_.resize(512,0);
  
//   //@@ below is temporary due to bug in SiStripConnections class!
//   std::vector<unsigned short>::iterator iter; 
//   for ( iter = feds.begin(); iter != feds.end(); iter++) {
//     bool new_id = true;
//     std::vector<unsigned short>::iterator ifed;
//     for ( ifed = fedids_.begin(); ifed != fedids_.end(); ifed++ ) {
//       //   if (*ifed == *iter) { new_id = false; break; }
//     }
//     if ( new_id ) { fedids_.push_back(*iter);}
//     }

//   //some debug
//   if (verbosity_>2) { 
//     std::cout << "[SiStripDigiToRaw::createDigis] "
// 	      << "Number of FED ids: " << fedids_.size() << ", "
// 	      << "List of FED ids: ";
//     for ( unsigned int ifed = 0; ifed < fedids_.size(); ifed++ ) { 
//       std::cout << fedids_[ifed] << ", ";
//     }
//     cout << std::endl;
//   }

//   //some debug
//   if (verbosity_>2) { 
//     std::map< unsigned short, std::vector<DetId> > partitions;
//     std::map< unsigned short, std::vector<DetId> >::iterator ifed;
//     connections_.getDetPartitions( partitions );
//     std::cout << "[SiStripDigiToRaw::SiStripDigiToRaw] "
// 	      << "Number of FED \"partitions\": " 
// 	      << partitions.size() << std::endl
// 	      << " FedId/nDets: ";
//     for ( ifed = partitions.begin(); ifed != partitions.end(); ifed++ ) { 
//       std::cout << (*ifed).first << "/"
// 		<< ((*ifed).second).size() << ", ";
//     }
//     cout << std::endl;
//   }

}

// -----------------------------------------------------------------------------
// destructor
SiStripDigiToRaw::~SiStripDigiToRaw() {
  std::cout << "[SiStripDigiToRaw::~SiStripDigiToRaw]" 
	    << " Destructing object..." << endl;

  // counters
  std::cout << "[SiStripDigiToRaw::~SiStripDigiToRaw] Some cumulative counters: "
	    << "#FEDs: " << nFeds_ 
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
void SiStripDigiToRaw::createFedBuffers( edm::ESHandle<SiStripReadoutCabling>& cabling,
					 edm::Handle<StripDigiCollection>& collection,
					 std::auto_ptr<FEDRawDataCollection>& buffers ) {

  if (verbosity_>2) std::cout << "[SiStripDigiToRaw::createFedBuffers] " << endl;

  try {
   
    // Some debug
    if (verbosity_>2) {
      std::pair<int,int> range = FEDNumbering::getSiStripFEDIds();
      std::vector<unsigned int> det_ids;
      collection->detIDs( det_ids );
      std::cout << "[SiStripDigiToRaw::createFedBuffers]" 
		<< " Number of FEDRawData objects is " 
		<< range.second - range.first
 		<< ", Number of detectors with digis: " 
 		<< det_ids.size() << std::endl;
    }

    // Define container for (raw) ADC values
    const unsigned short strips_per_fed = 96 * 256; 
    vector<unsigned short> data_buffer; 
    data_buffer.reserve(strips_per_fed);

    // Retrieve FED ids, iterate through FEDs and extract payload
    const std::vector<unsigned short> fed_ids = cabling->getFEDs();
    std::vector<unsigned short>::const_iterator ifed;
    for ( ifed = fed_ids.begin(); ifed != fed_ids.end(); ifed++ ) {
      
      if ( verbosity_ > 1 ) {
	std::cout << "[SiStripDigiToRaw::createFedBuffers]"
		  << " processing FED id " << *ifed << std::endl;
      }

      // Counter of FEDs for debug purposes
      nFeds_++; 

      // Initialise buffer holding ADC values
      data_buffer.clear();
      data_buffer.resize(strips_per_fed,0);

      //loop through FED channels
      for (unsigned short ichan = 0; ichan < 96; ichan++) {
	
	// Retrieve pair containing DetId and APV pair number
	pair<unsigned int, unsigned short> apv_pair_id = cabling->getAPVPair( *ifed, ichan );
	
	// Check DetId is non-zero
	if ( !apv_pair_id.first ) { 
	  if ( verbosity_ > 2 ) { 
	    std::cout  << "[SiStripDigiToRaw::createFedBuffers]"
		       << " Zero DetId returned for FED id " << *ifed
		       << " and channel " << ichan << std::endl; 
	  }
	  continue;
	}
	
	// Retrieve and iterate through Digis
	vector<StripDigi> digis;
	collection->digis( apv_pair_id.first, digis );
	if ( !digis.empty() ) cout << "size " << digis.size() << endl; 

	vector<StripDigi>::const_iterator idigi;
	for ( idigi = digis.begin(); idigi != digis.end(); idigi++ ) {
	  // Check strip is within range appropriate for this APV pair
	  
	  if ( (*idigi).strip() >= apv_pair_id.second*256 && 
	       (*idigi).strip() < (apv_pair_id.second + 1)*256 ) {
	    // Calc digi strip position, within scope of FED 
	    unsigned short strip = ichan*256 + (*idigi).strip()%256;
	    if ( strip >= strips_per_fed ) {
	      std::stringstream os;
	      os << "[SiStripDigiToRaw::createFedBuffers]"
		 << " strip >= strips_per_fed";
	      throw string( os.str() );
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
		std::cout << "Retrieved digi with ADC value " << (*idigi).adc()
			  << " from FED id " << *ifed 
			  << " and FED channel " << ichan
			  << " at detector strip position " << (*idigi).strip()
			  << " and FED strip position " << strip << std::endl;
	      }
	    }
	  }
	}
	// If strip greater than 
	//if ((*idigi).strip() >= (apv_pair_id.second + 1)*256) break;
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
      FEDRawData& fedrawdata = buffers->FEDData( *ifed ); 
      // calculate size of FED buffer in units of bytes (unsigned char)
      int nbytes = generator.getBufferSize() * 4;
      // resize (public) "data_" member of struct FEDRawData
      fedrawdata.resize( nbytes );
      // copy FED buffer to struct FEDRawData using Fed9UBufferGenerator
      unsigned char* chars = const_cast<unsigned char*>( fedrawdata.data() );
      unsigned int* ints = reinterpret_cast<unsigned int*>( chars );
      generator.getBuffer( ints );
      if ( creator ) { delete creator; }

    }
    
//     for ( ifed = fed_ids.begin(); ifed != fed_ids.end(); ifed++ ) {
//       std::cout << "[SiStripDigiToRaw::createFedBuffers]"
// 		<< " FED id " << *ifed
// 		<< " data size " << buffers->FEDData( *ifed ).size()
// 		<< std::endl; 
//     }

  }
  catch ( string err ) {
    std::cout << "SiStripDigiToRaw::createFedBuffers] " 
	      << "Exception caught : " << err << std::endl;
  }

  if (verbosity_>1) std::cout << "[SiStripDigiToRaw::createFedBuffers]" 
			      << " exiting method..." << std::endl;
  
}

