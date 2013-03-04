#include "EventFilter/SiStripRawToDigi/plugins/ExcludedFEDListProducer.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/src/fed_header.h"
#include "DataFormats/FEDRawData/src/fed_trailer.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripEventSummary.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "EventFilter/SiStripRawToDigi/interface/TFHeaderDescription.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <ext/algorithm>

namespace sistrip {

  ExcludedFEDListProducer::ExcludedFEDListProducer( const edm::ParameterSet& pset ) :
    runNumber_(0),
    productLabel_(pset.getParameter<edm::InputTag>("ProductLabel")),
    cabling_(0),
    cacheId_(0)
  {
    produces<DetIdCollection>();
  }
  
  ExcludedFEDListProducer::~ExcludedFEDListProducer()
  {
  }
  
  
  void ExcludedFEDListProducer::beginRun( const edm::Run & run, const edm::EventSetup & es)
  {
    uint32_t cacheId = es.get<SiStripFedCablingRcd>().cacheIdentifier();
    
    if ( cacheId_ != cacheId ) {
      
      edm::ESHandle<SiStripFedCabling> c;
      es.get<SiStripFedCablingRcd>().get( c );
      cabling_ = c.product();
    }
  }
  
  void ExcludedFEDListProducer::produce( edm::Event& event, const edm::EventSetup& es)
  {
    if( runNumber_ != event.run() ) {
    
      runNumber_ = event.run();

      DetIdCollection emptyDetIdCollection;
      detids_.swap(emptyDetIdCollection);
      // Reserve space in bad module list
      detids_.reserve(100);
    
      edm::Handle<FEDRawDataCollection> buffers;
      event.getByLabel( productLabel_, buffers ); 
 
      // Retrieve FED ids from cabling map and iterate through 
      std::vector<uint16_t>::const_iterator ifed = cabling_->feds().begin();
      for ( ; ifed != cabling_->feds().end(); ifed++ ) {
      
	// ignore trigger FED
	// if ( *ifed == triggerFedId_ ) { continue; }
	
	// Retrieve FED raw data for given FED 
	const FEDRawData& input = buffers->FEDData( static_cast<int>(*ifed) );
	// The FEDData contains a vector<unsigned char>. Check the size of this vector:
	
	if( input.size() == 0 ) {
	  // std::cout << "Input size == 0 for FED number " << static_cast<int>(*ifed) << std::endl;
	  // get the cabling connections for this FED
	  const std::vector<FedChannelConnection>& conns = cabling_->connections(*ifed);
	  // Mark FED modules as bad
	  detids_.reserve(detids_.size()+conns.size());
	  std::vector<FedChannelConnection>::const_iterator iconn = conns.begin();
	  for ( ; iconn != conns.end(); ++iconn ) {
	    if ( !iconn->detId() || iconn->detId() == sistrip::invalid32_ ) continue;
	    detids_.push_back(iconn->detId()); //@@ Possible multiple entries
	  }
	}
      }
    }

    std::auto_ptr< DetIdCollection > detids( new DetIdCollection(detids_) );

    // for( unsigned int it = 0; it < detids->size(); ++it ) {
    //   std::cout << "detId = " << (*detids)[it]() << std::endl;
    // }

    event.put(detids);
  }
}
