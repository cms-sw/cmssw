#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToDigiModule.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToDigi.h"
// handle to EventSetup
#include "FWCore/Framework/interface/ESHandle.h"
// data collections
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/SiStripDigi/interface/StripDigiCollection.h"
#include "DataFormats/SiStripDigi/interface/StripDigi.h"
// geometry 
#include "Geometry/TrackerSimAlgo/interface/CmsDigiTracker.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/DetId/interface/DetId.h"
// utility class
#include "EventFilter/SiStripRawToDigi/interface/SiStripUtility.h"
// connectivity
#include "CalibTracker/SiStripConnectivity/interface/SiStripConnection.h"
// #include "Geometry/Records/interface/TrackerConnectionRecord.h"
//
#include <cstdlib>

// -----------------------------------------------------------------------------
// constructor
SiStripRawToDigiModule::SiStripRawToDigiModule( const edm::ParameterSet& conf ) :
  rawToDigi_(0),
  utility_(0),
  event_(0),
  fedReadoutMode_( conf.getParameter<std::string>("FedReadoutMode") ),
  fedReadoutPath_( conf.getParameter<std::string>("FedReadoutPath") ),
  verbosity_( conf.getParameter<int>("Verbosity") ),
  edProductLabel_( conf.getParameter<std::string>("EDProductLabel") )
{
  if (verbosity_>1) std::cout << "[SiStripRawToDigiModule::SiStripRawToDigiModule] "
			      << "Constructing RawToDigi module..." << std::endl;

  // specify product type
  produces<StripDigiCollection>();
}

// -----------------------------------------------------------------------------
// destructor
SiStripRawToDigiModule::~SiStripRawToDigiModule() {
  if (verbosity_>1) std::cout << "[SiStripRawToDigiModule::~SiStripRawToDigiModule] "
			      << "Destructing RawToDigi module..." << std::endl;
  if ( rawToDigi_ ) delete rawToDigi_;
  if ( utility_ ) delete utility_; 
}

// -----------------------------------------------------------------------------
//
void SiStripRawToDigiModule::beginJob( const edm::EventSetup& iSetup ) {
  if (verbosity_>2) std::cout << "[SiStripRawToDigiModule::beginJob] "
			      << "creating utility object, connections map, "
			      << "RawToDigi converter..." << std::endl;
  
  //@@ cannot presently retrieve connections map from EventSetup!

  //   // retrieve cabling map (ESProduct) 
  //   ESHandle<SiStripConnection> connections;
  //   iSetup.get<TrackerConnectionRecord>().get( connections );
  //   cabling_.reset( connections.product() );
  
  // retrieve "dummy" connections map from utility object
  utility_ = new SiStripUtility( iSetup );
  if (verbosity_>1) utility_->verbose(true);
  SiStripConnection connections;
  utility_->siStripConnection( connections );
  
  // create instance of RawToDigi converter
  rawToDigi_ = new SiStripRawToDigi( connections, verbosity_ );
  rawToDigi_->fedReadoutPath( fedReadoutPath_ );
  rawToDigi_->fedReadoutMode( fedReadoutMode_ );
  
}

// -----------------------------------------------------------------------------
//
void SiStripRawToDigiModule::endJob() { 
  if (verbosity_>2) std::cout << "[SiStripRawToDigiModule::endJob] "
			      << "currently does nothing..." << std::endl;
}

// -----------------------------------------------------------------------------
// produces a StripDigiCollection
void SiStripRawToDigiModule::produce( edm::Event& iEvent, 
				      const edm::EventSetup& iSetup ) {
  if (verbosity_>2) std::cout << "[SiStripRawToDigiModule::produce] "
			      << "input: \"dummy\" StripDigiCollection, " 
			      << "output: FedRawDataCollection" << std::endl;
  
  event_++; // increment event counter

  if (verbosity_>0) std::cout << "[SiStripRawToDigiModule::produce] "
			      << "event number: " << event_ << std::endl;
  
  // retrieve collection of FEDRawData objects from Event
  edm::Handle<raw::FEDRawDataCollection> handle;
  iEvent.getByLabel(edProductLabel_, handle);
  raw::FEDRawDataCollection fed_buffers = const_cast<raw::FEDRawDataCollection&>( *handle );
  
  // // retrieve "dummy" collection of FEDRawData from utility object
  // raw::FEDRawDataCollection fed_buffers;
  // utility_->fedRawDataCollection( fed_buffers );
  
  // create product 
  std::auto_ptr<StripDigiCollection> digis( new StripDigiCollection );

  // use RawToDigi converter to fill FEDRawDataCollection
  rawToDigi_->createDigis( fed_buffers, *(digis.get()) );

  // write StripDigiCollection to the Event
  iEvent.put( digis );
  
}
