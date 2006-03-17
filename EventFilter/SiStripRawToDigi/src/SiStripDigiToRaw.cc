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
  anal_("SiStripDigiToRaw")
{
  cout << "[SiStripDigiToRaw::SiStripDigiToRaw]" 
       << " Constructing object..." << endl;
}

// -----------------------------------------------------------------------------
/** */
SiStripDigiToRaw::~SiStripDigiToRaw() {
  cout << "[SiStripDigiToRaw::~SiStripDigiToRaw]" 
       << " Destructing object..." << endl;
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
  anal_.addEvent();

  try {
    
    const uint16_t strips_per_fed = 96 * 256; 
    vector<uint16_t> raw_data; 
    raw_data.reserve(strips_per_fed);

    const vector<uint16_t>& fed_ids = cabling->feds();
    vector<uint16_t>::const_iterator ifed;
    for ( ifed = fed_ids.begin(); ifed != fed_ids.end(); ifed++ ) {
      anal_.addFed();
      cout << "[SiStripDigiToRaw::createFedBuffers]"
	   << " Processing FED id " << *ifed << endl;
      
      raw_data.clear(); raw_data.resize( strips_per_fed, 0 );

      for ( uint16_t ichan = 0; ichan < 96; ichan++ ) {
	anal_.addChan();
	
	const FedChannelConnection& conn = cabling->connection( *ifed, ichan );
	//@@ Check DetId is non-zero?
	vector< edm::DetSet<SiStripDigi> >::const_iterator digis = collection->find( conn.detId() );
	if ( digis->data.empty() ) { cout << "[SiStripDigiToRaw::createFedBuffers]"
					  << " Zero digis found!" << endl; }
	
	edm::DetSet<SiStripDigi>::const_iterator idigi;
	for ( idigi = digis->data.begin(); idigi != digis->data.end(); idigi++ ) {
	  if ( (*idigi).strip() < conn.apvPairNumber()*256 ||
	       (*idigi).strip() > conn.apvPairNumber()*256+255 ) { continue; }
//  	  if ( (*idigi).strip() >= conn.apvPairNumber()*256 && 
//  	       (*idigi).strip() < (conn.apvPairNumber()+1)*256 ) {
	    unsigned short strip = ichan*256 + (*idigi).strip()%256;
	    if ( strip >= strips_per_fed ) {
	      stringstream os;
	      os << "[SiStripDigiToRaw::createFedBuffers]"
		 << " strip >= strips_per_fed";
	      throw string( os.str() );
	    }
	    cout << "[SiStripDigiToRaw::createFedBuffers]"
		 << " Retrieved digi!"
		 << "  AdcValue: " << (*idigi).adc()
		 << "  FedId/FedCh: " << *ifed << "/" << ichan
		 << "  DetStrip: " << (*idigi).strip()
		 << "  FedStrip: " << strip << endl;
	    // check if value already exists
	    if ( raw_data[strip] && raw_data[strip] != (*idigi).adc() ) {
		stringstream os; 
		os << "[SiStripDigiToRaw::createFedBuffers]" 
		   << " Incompatible ADC values in buffer!"
		   << "  FedId/FedCh: " << *ifed << "/" << ichan
		   << "  DetStrip: " << (*idigi).strip() 
		   << "  FedStrip: " << strip
		   << "  AdcValue: " << (*idigi).adc()
		   << "  RawData[" << strip << "]: " << raw_data[strip];
		cerr << os.str() << endl;
	      }
	    // Add digi to buffer
	    raw_data[strip] = (*idigi).adc();
	    if ( readoutMode_ == "SCOPE_MODE" ) { 
	      anal_.smDigi( (*idigi).strip(), (*idigi).adc() ); 
	    } else if ( readoutMode_ == "VIRGIN_RAW" ) {
	      anal_.vrDigi( (*idigi).strip(), (*idigi).adc() ); 
	    } else if ( readoutMode_ == "PROCESSED_RAW" ) {
	      anal_.prDigi( (*idigi).strip(), (*idigi).adc() ); 
	    } else if ( readoutMode_ == "ZERO_SUPPRESSED" ) {
	      anal_.zsDigi( (*idigi).strip(), (*idigi).adc() ); 
	    }
	}
	// if ((*idigi).strip() >= (conn.apvPairNumber()+1)*256) break;
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

