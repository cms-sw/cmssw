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
  fedReadoutPath_( conf.getParameter<string>("FedReadoutPath") ),
  verbosity_( conf.getParameter<int>("Verbosity") )
{
  if (verbosity_>1) std::cout << "[SiStripRawToDigiModule] "
			      << "constructing RawToDigi module..." << endl;
  // specify product type
  produces<StripDigiCollection>("StripDigiCollection_from_FEDRawData");
}

// -----------------------------------------------------------------------------
// destructor
SiStripRawToDigiModule::~SiStripRawToDigiModule() {
  if (verbosity_>1) std::cout << "[SiStripRawToDigiModule] "
			      << "destructing RawToDigi module..." << endl;
  if ( rawToDigi_ ) delete rawToDigi_;
  if ( utility_ ) delete utility_; 
}

// -----------------------------------------------------------------------------
//
void SiStripRawToDigiModule::beginJob( const edm::EventSetup& iSetup ) {
  if (verbosity_>2) std::cout << "[SiStripRawToDigiModule::beginJob] "
			      << "creating utility object, connections map, RawToDigi converter..." << endl;
  
  //@@ cannot presently retrieve connections map from EventSetup!
  //   // retrieve cabling map (ESProduct) 
  //   ESHandle<SiStripConnection> connections;
  //   iSetup.get<TrackerConnectionRecord>().get( connections );
  //   cabling_.reset( connections.product() );
  
  // retrieve "dummy" connections map from utility object
  utility_ = new SiStripUtility( iSetup );
  SiStripConnection connections;
  utility_->siStripConnection( connections );
  
  // some debug:
  vector<unsigned short> feds;
  map<unsigned short, cms::DetId> partitions;
  connections.getConnectedFedNumbers( feds );
  connections.getDetPartitions( partitions );
  std::cout << "number of feds: " << feds.size() 
	    << ", number of partitions: " << partitions.size() << endl;
  
  // create instance of RawToDigi converter
  rawToDigi_ = new SiStripRawToDigi( connections );
  // rawToDigi_->fedReadoutPath( fedReadoutPath_ );
  
}

// -----------------------------------------------------------------------------
//
void SiStripRawToDigiModule::endJob() { 
  if (verbosity_>2) std::cout << "[SiStripRawToDigiModule::endJob] "
			      << "cuurently does nothing..." << endl;
}

// -----------------------------------------------------------------------------
// produces a FEDRawDataCollection
void SiStripRawToDigiModule::produce( edm::Event& iEvent, 
				      const edm::EventSetup& iSetup ) {
  if (verbosity_>2) std::cout << "[SiStripRawToDigiModule::produce] "
			      << "creates \"dummy\" FedRawDataCollection as " << endl
			      << "input to RawToDigi converter and writes "
			      << "\"StripDigiCollection\" product to Event ..." << endl;
  
  event_++; // increment event counter
  if (verbosity_>0) std::cout << "[SiStripRawToDigiModule::produce] "
			      << "event number: " << event_ << endl;
  
  //   // retrieve collection of StripDigi's from Event
  //   edm::Handle<StripDigiCollection> input;
  //   e.getByLabel("StripDigiConverter", input);
  
  // retrieve "dummy" collection of FEDRawData from utility object
  raw::FEDRawDataCollection fed_buffers;
  utility_->fedRawDataCollection( fed_buffers );
  
  // some debug:
  for ( int ifed = 0; ifed < 1023; ifed++ ) {
    raw::FEDRawData& data = fed_buffers.FEDData( ifed );
    if ( !(data.data_).size() ) { cout << "FED " << ifed << " has an empty FEDRawData!" << endl; }
    else { cout << "FED " << ifed << " has a FEDRawData with size " << (data.data_).size() << endl; }
  }

  // create product 
  StripDigiCollection digis;

  // use RawToDigi converter to fill FEDRawDataCollection
  rawToDigi_->createDigis( fed_buffers, digis );

  // write StripDigiCollection to the Event
  // iEvent.put( &digis );

}

// -----------------------------------------------------------------------------
// define the class SiStripRawToDigiModule as a plug-in
//DEFINE_FWK_MODULE(SiStripRawToDigiModule)
