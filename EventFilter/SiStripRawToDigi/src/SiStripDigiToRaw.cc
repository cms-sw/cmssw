#include "EventFilter/SiStripRawToDigi/interface/SiStripDigiToRaw.h"
// timing
#include "Utilities/Timing/interface/TimingReport.h"
// data formats
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
// cabling
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
// std
#include <iostream>
#include <sstream>
#include <vector>

// -----------------------------------------------------------------------------
/** */
SiStripDigiToRaw::SiStripDigiToRaw( string mode, int16_t nbytes ) : 
  readoutMode_(mode),
  nAppendedBytes_(nbytes),
  position_(), 
  landau_(),
  nFeds_(0), 
  nDigis_(0)
{
  cout << "[SiStripDigiToRaw::SiStripDigiToRaw]" 
       << " Constructing object..." << endl;
  
  landau_.clear(); landau_.reserve(1024); landau_.resize(1024,0);
  position_.clear(); position_.reserve(768); position_.resize(768,0);

}

// -----------------------------------------------------------------------------
/** */
SiStripDigiToRaw::~SiStripDigiToRaw() {
  cout << "[SiStripDigiToRaw::~SiStripDigiToRaw]" 
	    << " Destructing object..." << endl;

  cout << "[SiStripDigiToRaw::~SiStripDigiToRaw]"
       << " Some cumulative counters: nFeds_: " << nFeds_ 
       << " nDigis_: " << nDigis_ << endl;
  
  cout << "[SiStripDigiToRaw::~SiStripDigiToRaw] "
       << "Digi statistics (vs strip position): " << endl;
  int tmp1 = 0;
  for ( uint16_t i = 0; i < position_.size(); i++ ) {
    if ( i<10 ) { cout << "Strip: " << i << ",  Digis: " << position_[i] << endl; }
    tmp1 += position_[i];
  }
  cout << "nDigis: " << tmp1 << endl;

  cout << "[SiStripDigiToRaw::~SiStripDigiToRaw]"
       << " Landau statistics: " << endl;
  int tmp2 = 0;
  for ( uint16_t i = 0; i < landau_.size(); i++ ) {
    if ( i<10 ) { cout << "ADC: " << i << ",  Digis: " << landau_[i] << endl; }
    tmp2 += landau_[i];
  }
  cout << "nDigis: " << tmp2 << endl;

}

// -----------------------------------------------------------------------------
/** 
    Input: DetSetVector of SiStripDigis. Output: FEDRawDataCollection.
//     Retrieves and iterates through FED buffers, extract FEDRawData
//     from collection and (optionally) dumps raw data to stdout, locates
//     start of FED buffer by identifying DAQ header, creates new
//     Fed9UEvent object using current FEDRawData buffer, dumps FED
//     buffer to stdout, retrieves data from various header fields
*/
void SiStripDigiToRaw::createFedBuffers( edm::ESHandle<SiStripFedCabling>& cabling,
					 edm::Handle< edm::DetSetVector<SiStripDigi> >& collection,
					 auto_ptr<FEDRawDataCollection>& buffers ) {
  cout << "[SiStripDigiToRaw::createFedBuffers] " << endl;

  try {
    
    const uint16_t strips_per_fed = 96 * 256; 
    vector<uint16_t> raw_data; 
    raw_data.reserve(strips_per_fed);

    const vector<uint16_t>& fed_ids = cabling->feds();
    vector<uint16_t>::const_iterator ifed;
    for ( ifed = fed_ids.begin(); ifed != fed_ids.end(); ifed++ ) {
      
      cout << "[SiStripDigiToRaw::createFedBuffers]"
	   << " Processing FED id " << *ifed << endl;
      nFeds_++; 
      
      raw_data.clear(); raw_data.resize( strips_per_fed, 0 );

      for ( uint16_t ichan = 0; ichan < 96; ichan++ ) {
	
	const FedChannelConnection& conn = cabling->connection( *ifed, ichan );
	//@@ Check DetId is non-zero?
	vector< edm::DetSet<SiStripDigi> >::const_iterator digis = collection->find( conn.detId() );
	if ( digis->data.empty() ) { cout << "[SiStripDigiToRaw::createFedBuffers]"
					  << " Zero digis found!" << endl; }
	
	edm::DetSet<SiStripDigi>::const_iterator idigi;
	for ( idigi = digis->data.begin(); idigi != digis->data.end(); idigi++ ) {
	  
	  if ( (*idigi).strip() >= conn.pairId()*256 && 
	       (*idigi).strip() < (conn.pairId()+1)*256 ) {
	    unsigned short strip = ichan*256 + (*idigi).strip()%256;
	    if ( strip >= strips_per_fed ) {
	      stringstream os;
	      os << "[SiStripDigiToRaw::createFedBuffers]"
		 << " strip >= strips_per_fed";
	      throw string( os.str() );
	    }
	    // check if buffer has already been filled with digi ADC value. 
	    // if not or if filled with different value, fill it.
	    if ( raw_data[strip] ) { // if yes, cross-check values
	      if ( raw_data[strip] != (*idigi).adc() ) {
		stringstream os; 
		os << "SiStripDigiToRaw::createFedBuffers(.): " 
		   << "WARNING: Incompatible ADC values in buffer: "
		   << "FED id: " << *ifed << ", FED channel: " << ichan
		   << ", detector strip: " << (*idigi).strip() 
		   << ", FED strip: " << strip
		   << ", ADC value: " << (*idigi).adc()
		   << ", raw_data["<<strip<<"]: " << raw_data[strip];
		cout << os.str() << endl;
	      }
	    } else { // if no, update buffer with digi ADC value
	      raw_data[strip] = (*idigi).adc(); 
	      // debug: update counters
	      position_[ (*idigi).strip() ]++;
	      landau_[ (*idigi).adc()<100 ? (*idigi).adc() : 0 ]++;
	      nDigis_++;
	      cout << "Retrieved digi with ADC value " << (*idigi).adc()
		   << " from FED id " << *ifed 
		   << " and FED channel " << ichan
		   << " at detector strip position " << (*idigi).strip()
		   << " and FED strip position " << strip << endl;
	    }
	  }
	}
	// If strip greater than 
	//if ((*idigi).strip() >= (conn.pairId()+1)*256) break;
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
	cout << "WARNING : UNKNOWN readout mode" << endl;
      }
    
      // generate FED buffer and pass to Daq
      Fed9U::Fed9UBufferGenerator generator( creator );
      generator.generateFed9UBuffer( raw_data );
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
    
  }
  catch ( string err ) {
    cout << "SiStripDigiToRaw::createFedBuffers] " 
	 << "Exception caught : " << err << endl;
  }
  
}

