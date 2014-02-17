// Last commit: $Id: SiStripDigiToRaw.cc,v 1.41 2009/09/14 14:01:04 nc302 Exp $

#include "EventFilter/SiStripRawToDigi/interface/SiStripDigiToRaw.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/Utilities/interface/CRC16.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>
#include <vector>

namespace sistrip {

  // -----------------------------------------------------------------------------
  /** */
  DigiToRaw::DigiToRaw( FEDReadoutMode mode, 
			bool useFedKey ) : 
    mode_(mode),
    useFedKey_(useFedKey),
    bufferGenerator_()
  {
    if ( edm::isDebugEnabled() ) {
      LogDebug("DigiToRaw")
	<< "[sistrip::DigiToRaw::DigiToRaw]"
	<< " Constructing object...";
    }
    bufferGenerator_.setReadoutMode(mode_);
  }

  // -----------------------------------------------------------------------------
  /** */
  DigiToRaw::~DigiToRaw() {
    if ( edm::isDebugEnabled() ) {
      LogDebug("DigiToRaw")
	<< "[sistrip::DigiToRaw::~DigiToRaw]"
	<< " Destructing object...";
    }
  }

  // -----------------------------------------------------------------------------
  /** 
      Input: DetSetVector of SiStripDigis. Output: FEDRawDataCollection.
      Retrieves and iterates through FED buffers, extract FEDRawData
      from collection and (optionally) dumps raw data to stdout, locates
      start of FED buffer by identifying DAQ header, creates new
      Fed9UEvent object using current FEDRawData buffer, dumps FED
      buffer to stdout, retrieves data from various header fields
  */
  void DigiToRaw::createFedBuffers( edm::Event& event,
				    edm::ESHandle<SiStripFedCabling>& cabling,
				    edm::Handle< edm::DetSetVector<SiStripDigi> >& collection,
				    std::auto_ptr<FEDRawDataCollection>& buffers ) { 
    createFedBuffers_(event, cabling, collection, buffers, true);
  }
  
  void DigiToRaw::createFedBuffers( edm::Event& event,
				    edm::ESHandle<SiStripFedCabling>& cabling,
				    edm::Handle< edm::DetSetVector<SiStripRawDigi> >& collection,
				    std::auto_ptr<FEDRawDataCollection>& buffers ) { 
    createFedBuffers_(event, cabling, collection, buffers, false);
  }
  
  template<class Digi_t>
  void DigiToRaw::createFedBuffers_( edm::Event& event,
				     edm::ESHandle<SiStripFedCabling>& cabling,
				     edm::Handle< edm::DetSetVector<Digi_t> >& collection,
				     std::auto_ptr<FEDRawDataCollection>& buffers,
				     bool zeroSuppressed) {
    try {
      
      //set the L1ID to use in the buffers
      bufferGenerator_.setL1ID(0xFFFFFF & event.id().event());
      
      const std::vector<uint16_t>& fed_ids = cabling->feds();
      std::vector<uint16_t>::const_iterator ifed;
      
      for ( ifed = fed_ids.begin(); ifed != fed_ids.end(); ifed++ ) {
        
        const std::vector<FedChannelConnection>& conns = cabling->connections(*ifed);
	std::vector<FedChannelConnection>::const_iterator iconn = conns.begin();
        
        FEDStripData fedData(zeroSuppressed);
        
	for ( ; iconn != conns.end(); iconn++ ) {
          
	  // Determine FED key from cabling
	  uint32_t fed_key = ( ( iconn->fedId() & sistrip::invalid_ ) << 16 ) | ( iconn->fedCh() & sistrip::invalid_ );
	
	  // Determine whether DetId or FED key should be used to index digi containers
	  uint32_t key = ( useFedKey_ || mode_ == READOUT_MODE_SCOPE ) ? fed_key : iconn->detId();
          
          // Check key is non-zero and valid
	  if ( !key || ( key == sistrip::invalid32_ ) ) { continue; }

	  // Determine APV pair number (needed only when using DetId)
	  uint16_t ipair = ( useFedKey_ || mode_ == READOUT_MODE_SCOPE ) ? 0 : iconn->apvPairNumber();
          
          FEDStripData::ChannelData& chanData = fedData[iconn->fedCh()];

	  // Find digis for DetID in collection
	  typename std::vector< edm::DetSet<Digi_t> >::const_iterator digis = collection->find( key );
	  if (digis == collection->end()) { continue; } 

	  typename edm::DetSet<Digi_t>::const_iterator idigi, digis_begin(digis->data.begin());
	  for ( idigi = digis_begin; idigi != digis->data.end(); idigi++ ) {
	    
	    if ( STRIP(idigi, digis_begin) < ipair*256 ||
		 STRIP(idigi, digis_begin) > ipair*256+255 ) { continue; }
	    const unsigned short strip = STRIP(idigi, digis_begin) % 256;

	    if ( strip >= STRIPS_PER_FEDCH ) {
	      if ( edm::isDebugEnabled() ) {
		std::stringstream ss;
		ss << "[sistrip::DigiToRaw::createFedBuffers]"
		   << " strip >= strips_per_fedCh";
		edm::LogWarning("DigiToRaw") << ss.str();
	      }
	      continue;
	    }
	  
	    // check if value already exists
	    if ( edm::isDebugEnabled() ) {
              const uint16_t value = 0;//chanData[strip];
	      if ( value && value != (*idigi).adc() ) {
		std::stringstream ss; 
		ss << "[sistrip::DigiToRaw::createFedBuffers]" 
		   << " Incompatible ADC values in buffer!"
		   << "  FedId/FedCh: " << *ifed << "/" << iconn->fedCh()
		   << "  DetStrip: " << STRIP(idigi, digis_begin)
		   << "  FedChStrip: " << strip
		   << "  AdcValue: " << (*idigi).adc()
		   << "  RawData[" << strip << "]: " << value;
		edm::LogWarning("DigiToRaw") << ss.str();
	      }
	    }

	    // Add digi to buffer
	    chanData[strip] = (*idigi).adc();

	  }
	  // if ((*idigi).strip() >= (ipair+1)*256) break;
	}

        //create the buffer
        FEDRawData& fedrawdata = buffers->FEDData( *ifed );
        bufferGenerator_.generateBuffer(&fedrawdata,fedData,*ifed);
        
       }
    }
    catch (const std::exception& e) {
      if ( edm::isDebugEnabled() ) {
	edm::LogWarning("DigiToRaw") 
	  << "DigiToRaw::createFedBuffers] " 
	  << "Exception caught : " << e.what();
      }
    }
  
  }

  inline uint16_t DigiToRaw::STRIP(const edm::DetSet<SiStripDigi>::const_iterator& it, const edm::DetSet<SiStripDigi>::const_iterator& begin) const {return it->strip();}
  inline uint16_t DigiToRaw::STRIP(const edm::DetSet<SiStripRawDigi>::const_iterator& it, const edm::DetSet<SiStripRawDigi>::const_iterator& begin) const {return it-begin;}

}

