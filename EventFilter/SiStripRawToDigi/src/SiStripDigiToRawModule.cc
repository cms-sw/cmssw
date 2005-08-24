#include "EventFilter/SiStripRawToDigi/interface/SiStripDigiToRawModule.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripDigiToRaw.h"
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
// utility class
#include "EventFilter/SiStripRawToDigi/interface/SiStripUtility.h"
// connectivity
#include "CalibTracker/SiStripConnectivity/interface/SiStripConnection.h"
// #include "Geometry/Records/interface/TrackerConnectionRecord.h"
//
#include <cstdlib>

// -----------------------------------------------------------------------------
// constructor
SiStripDigiToRawModule::SiStripDigiToRawModule( const edm::ParameterSet& conf ) :
  digiToRaw_(0),
  utility_(0),
  event_(0),
  fedReadoutMode_( conf.getParameter<std::string>("FedReadoutMode") ),
  fedReadoutPath_( conf.getParameter<std::string>("FedReadoutPath") ),
  verbosity_( conf.getParameter<int>("Verbosity") ),
  ndigis_(0)
{
  if (verbosity_>1) std::cout << "[SiStripDigiToRawModule::SiStripDigiToRawModule] "
			      << "constructing DigiToRaw module..." << std::endl;
  // specify product type
  produces<raw::FEDRawDataCollection>();
}

// -----------------------------------------------------------------------------
// destructor
SiStripDigiToRawModule::~SiStripDigiToRawModule() {
  if (verbosity_>1) std::cout << "[SiStripDigiToRawModule::~SiStripDigiToRawModule] "
			      << "destructing DigiToRaw module..." << std::endl;
  if ( digiToRaw_ ) delete digiToRaw_;
  if ( utility_ ) delete utility_; 
  
  if (verbosity_>2) std::cout << "[SiStripDigiToRawModule::~SiStripDigiToRawModule] "
			      << "Total number of digis: " << ndigis_ << std::endl;
  
}

// -----------------------------------------------------------------------------
//
void SiStripDigiToRawModule::beginJob( const edm::EventSetup& iSetup ) {
  if (verbosity_>2) std::cout << "[SiStripDigiToRawModule::beginJob] "
			      << "creating utility object, connections map, "
			      << "DigiToRaw converter..." << std::endl;

  //   // retrieve cabling map (ESProduct) 
  //   ESHandle<SiStripConnection> connections;
  //   iSetup.get<TrackerConnectionRecord>().get( connections );
  //   cabling_.reset( connections.product() );

  //@@ cannot presently retrieve connections map from EventSetup, so use below!
  
  // create utility object and retrieve "dummy" connections map
  utility_ = new SiStripUtility( iSetup );
  if (verbosity_>1) utility_->verbose(true);
  SiStripConnection connections;
  utility_->siStripConnection( connections );
  
  // create instance of DigiToRaw converter
  digiToRaw_ = new SiStripDigiToRaw( connections );
  digiToRaw_->fedReadoutPath( fedReadoutPath_ );
  digiToRaw_->fedReadoutMode( fedReadoutMode_ );

}

// -----------------------------------------------------------------------------
//
void SiStripDigiToRawModule::endJob() { 
  if (verbosity_>2) std::cout << "[SiStripDigiToRawModule::endJob] "
			      << "cuurently does nothing..." << std::endl;
}

// -----------------------------------------------------------------------------
// produces a FEDRawDataCollection
void SiStripDigiToRawModule::produce( edm::Event& iEvent, 
				      const edm::EventSetup& iSetup ) {
  if (verbosity_>2) std::cout << "[SiStripDigiToRawModule::produce] "
			      << "creates \"dummy\" StripDigiCollection as " << std::endl
			      << "input to DigiToRaw converter and writes "
			      << "\"FedRawDataCollection\" product to Event ..." << std::endl;
  
  event_++; // increment event counter

  if (verbosity_>0) std::cout << "[SiStripDigiToRawModule::produce] "
			      << "event number: " << event_ << std::endl;
  
  //   // retrieve collection of StripDigi's from Event
  //   edm::Handle<StripDigiCollection> handle;
  //   //iEvent.getByLabel( "write_digis", handle );
  //   iEvent.getByLabel( "RawToDigi", handle );
  //   StripDigiCollection digis = const_cast<StripDigiCollection&>( *handle );

  //@@ cannot retrieve digis from Event based on simulated events, so use below!
  
  // retrieve "dummy" collection of StripDigi from utility object
  StripDigiCollection digis;
  utility_->stripDigiCollection( digis );
  
  // count digis
  std::vector<unsigned int> dets = digis.detIDs();
  for ( unsigned int idet = 0; idet < dets.size(); idet++ ) {
    const StripDigiCollection::Range digi_range = digis.get( idet ); 
    StripDigiCollection::ContainerIterator idigi;
    for ( idigi = digi_range.first; idigi != digi_range.second; idigi++ ) { ndigis_++; }
  }
  
  // create product
  std::auto_ptr<raw::FEDRawDataCollection> fed_buffers( new raw::FEDRawDataCollection );

  // use DigiToRaw converter to fill FEDRawDataCollection
  digiToRaw_->createFedBuffers( digis, *(fed_buffers.get()) );

  // write FEDRawDataCollection to the Event
  iEvent.put( fed_buffers );

}
