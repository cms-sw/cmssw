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
  //fedReadoutMode_( conf.getParameter<string>("FedReadoutMode") ),
  fedReadoutPath_( conf.getParameter<string>("FedReadoutPath") ),
  verbosity_( conf.getParameter<int>("Verbosity") ),
  ndigis_(0)
{
  if (verbosity_>1) std::cout << "[SiStripRawToDigiModule::SiStripRawToDigiModule] "
			      << "constructing RawToDigi module..." << endl;
  // specify product type
  produces<StripDigiCollection>();
}

// -----------------------------------------------------------------------------
// destructor
SiStripRawToDigiModule::~SiStripRawToDigiModule() {
  if (verbosity_>1) std::cout << "[SiStripRawToDigiModule::~SiStripRawToDigiModule] "
			      << "destructing RawToDigi module..." << endl;
  if ( rawToDigi_ ) delete rawToDigi_;
  if ( utility_ ) delete utility_; 
  
  std::cout << "[SiStripRawToDigiModule::~SiStripRawToDigiModule] Total number of digis: " << ndigis_ << endl;

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
  
//   // some debug
//   vector<unsigned short> feds;
//   map<unsigned short, cms::DetId> partitions;
//   connections.getConnectedFedNumbers( feds );
//   connections.getDetPartitions( partitions );
//   std::cout << "number of feds: " << feds.size() 
// 	    << ", number of partitions: " << partitions.size() << endl;
  
  // create instance of RawToDigi converter
  rawToDigi_ = new SiStripRawToDigi( connections );
  // rawToDigi_->fedReadoutPath( fedReadoutPath_ );
  // rawToDigi_->fedReadoutMode( fedReadoutMode_ );
  
}

// -----------------------------------------------------------------------------
//
void SiStripRawToDigiModule::endJob() { 
  if (verbosity_>2) std::cout << "[SiStripRawToDigiModule::endJob] "
			      << "cuurently does nothing..." << endl;
}

// -----------------------------------------------------------------------------
// produces a StripDigiCollection
void SiStripRawToDigiModule::produce( edm::Event& iEvent, 
				      const edm::EventSetup& iSetup ) {
  if (verbosity_>2) std::cout << "[SiStripRawToDigiModule::produce] "
			      << "creates \"dummy\" FedRawDataCollection as " << endl
			      << "input to RawToDigi converter and writes "
			      << "\"StripDigiCollection\" product to Event ..." << endl;
  
  event_++; // increment event counter
  if (verbosity_>0) std::cout << "[SiStripRawToDigiModule::produce] "
			      << "event number: " << event_ << endl;
  
  // retrieve collection of FEDRawData objects from Event
  edm::Handle<raw::FEDRawDataCollection> handle;
  //iEvent.getByLabel("DaqRawData", handle);
  iEvent.getByLabel("DigiToRaw", handle);
  raw::FEDRawDataCollection fed_buffers = const_cast<raw::FEDRawDataCollection&>( *handle );

  //   // retrieve "dummy" collection of FEDRawData from utility object
  //   raw::FEDRawDataCollection fed_buffers;
  //   utility_->fedRawDataCollection( fed_buffers );

  // some debug
  if (verbosity_>2) {
    int filled = 0;
    for ( int ifed = 0; ifed < 1023; ifed++ ) {
      if ( ( fed_buffers.FEDData(ifed) ).data_.size() ) { filled++; } 
    }
    std::cout << "[SiStripRawToDigiModule::produce] number of FEDRawData objects is " << filled << endl;
  }

  // create product 
  std::auto_ptr<StripDigiCollection> digis( new StripDigiCollection );

  // use RawToDigi converter to fill FEDRawDataCollection
  rawToDigi_->createDigis( fed_buffers, *(digis.get()) );

  // count number of digis
  std::vector<unsigned int> dets = digis->detIDs();
  for ( unsigned int idet = 0; idet < dets.size(); idet++ ) {
    const StripDigiCollection::Range digi_range = digis->get( idet ); 
    StripDigiCollection::ContainerIterator idigi;
    int ndigi = 0;
    for ( idigi = digi_range.first; idigi != digi_range.second; idigi++ ) { ndigis_++; ndigi++; }
    //if ( !ndigi ) { std::cout << "DET " << idet << " has zero digis!" << endl; }
    //else { std::cout << "DET " << idet << " has " << ndigi << " digis" << endl; }
  }

  // write StripDigiCollection to the Event
  iEvent.put( digis );
  
}
