// Last commit: $Id: SiStripDigiToRaw.cc,v 1.16 2007/04/24 16:58:59 bainbrid Exp $

#include "EventFilter/SiStripRawToDigi/interface/SiStripDigiToRaw.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Utilities/Timing/interface/TimingReport.h"
#include "Fed9UUtils.hh"
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

// -----------------------------------------------------------------------------
/** */
SiStripDigiToRaw::SiStripDigiToRaw( std::string mode, int16_t nbytes ) : 
  readoutMode_(mode),
  nAppendedBytes_(nbytes)
{
  LogDebug("DigiToRaw") << "[SiStripDigiToRaw::SiStripDigiToRaw] Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
SiStripDigiToRaw::~SiStripDigiToRaw() {
  LogDebug("DigiToRaw") << "[SiStripDigiToRaw::~SiStripDigiToRaw] Destructing object...";
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

  try {
    
    const uint16_t strips_per_fed = 96 * 256; 
    vector<uint16_t> raw_data; 
    raw_data.reserve(strips_per_fed);
    
    const vector<uint16_t>& fed_ids = cabling->feds();
    vector<uint16_t>::const_iterator ifed;

    for ( ifed = fed_ids.begin(); ifed != fed_ids.end(); ifed++ ) {

      LogDebug("DigiToRaw") << "[SiStripDigiToRaw::createFedBuffers] Processing FED id " << *ifed;
      
      raw_data.clear(); raw_data.resize( strips_per_fed, 0 );

      for ( uint16_t ichan = 0; ichan < 96; ichan++ ) {

	const FedChannelConnection& conn = cabling->connection( *ifed, ichan );

	// Check DetID is non-zero and valid
	if (!conn.detId() || 
	    (conn.detId()==sistrip::invalid_)) { continue; } 

	vector< edm::DetSet<SiStripDigi> >::const_iterator digis = collection->find( conn.detId() );

	// Check for Det Id in DetSetVector
	if (digis == collection->end()) { continue; } 

	/*
	if ( digis->data.empty() ) { 
	  edm::LogWarning("DigiToRaw") << "[SiStripDigiToRaw::createFedBuffers] Zero digis found!"; 
	}
	*/

	edm::DetSet<SiStripDigi>::const_iterator idigi;
	for ( idigi = digis->data.begin(); idigi != digis->data.end(); idigi++ ) {
	  if ( (*idigi).strip() < conn.apvPairNumber()*256 ||
	       (*idigi).strip() > conn.apvPairNumber()*256+255 ) { continue; }
	  unsigned short strip = ichan*256 + (*idigi).strip()%256;
	  if ( strip >= strips_per_fed ) {
	    std::stringstream ss;
	    ss << "[SiStripDigiToRaw::createFedBuffers]"
	       << " strip >= strips_per_fed";
	    edm::LogWarning("DigiToRaw") << ss.str();
	  }
	  LogDebug("DigiToRaw") << "[SiStripDigiToRaw::createFedBuffers]"
				<< " Retrieved digi!"
				<< "  AdcValue: " << (*idigi).adc()
				<< "  FedId/FedCh: " << *ifed << "/" << ichan
				<< "  DetStrip: " << (*idigi).strip()
				<< "  FedStrip: " << strip;
	  // check if value already exists
	  if ( raw_data[strip] && raw_data[strip] != (*idigi).adc() ) {
	    std::stringstream ss; 
	    ss << "[SiStripDigiToRaw::createFedBuffers]" 
	       << " Incompatible ADC values in buffer!"
	       << "  FedId/FedCh: " << *ifed << "/" << ichan
	       << "  DetStrip: " << (*idigi).strip() 
	       << "  FedStrip: " << strip
	       << "  AdcValue: " << (*idigi).adc()
	       << "  RawData[" << strip << "]: " << raw_data[strip];
	    edm::LogWarning("DigiToRaw") << ss.str();
	  }
	  // Add digi to buffer
	  raw_data[strip] = (*idigi).adc();
	}
	// if ((*idigi).strip() >= (conn.apvPairNumber()+1)*256) break;
      }

      // instantiate appropriate buffer creator object depending on readout mode
      Fed9U::Fed9UBufferCreator* creator = 0;
      if ( readoutMode_ == "SCOPE_MODE" ) {
	edm::LogWarning("DigiToRaw") << "Fed9UBufferCreatorScopeMode not implemented yet!";
      } else if ( readoutMode_ == "VIRGIN_RAW" ) {
	creator = new Fed9U::Fed9UBufferCreatorRaw();
      } else if ( readoutMode_ == "PROCESSED_RAW" ) {
	creator = new Fed9U::Fed9UBufferCreatorProcRaw();
      } else if ( readoutMode_ == "ZERO_SUPPRESSED" ) {
	creator = new Fed9U::Fed9UBufferCreatorZS();
      } else {
	edm::LogWarning("DigiToRaw") << "UNKNOWN readout mode";
      }
  
      // generate FED buffer and pass to Daq
      Fed9U::Fed9UBufferGenerator generator( creator );
      generator.generateFed9UBuffer( raw_data );
      //generator.setSlink64();
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
  catch ( std::string err ) {
    edm::LogWarning("DigiToRaw") << "SiStripDigiToRaw::createFedBuffers] " 
				 << "Exception caught : " << err;
  }
  
}

