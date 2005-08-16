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
// #include "CalibTracker/SiStripConnectivity/interface/SiStripConnection.h"
// #include "Geometry/Records/interface/TrackerConnectionRecord.h"
//
#include <cstdlib>

using namespace std;
using namespace raw;

// -----------------------------------------------------------------------------
// constructor
SiStripDigiToRawModule::SiStripDigiToRawModule( const edm::ParameterSet& conf ) :
  digiToRaw_(0),
  utility_(0),
  event_(0),
  fedReadoutPath_( conf.getParameter<string>("FedReadoutPath") ),
  verbosity_( conf.getParameter<int>("Verbosity") )
{
  if (verbosity_>1) cout << "[SiStripDigiToRawModule] "
			 << "constructing DigiToRaw module..." << endl;
  // specify product type
  produces<FEDRawDataCollection>("FEDRawDataCollection_from_StripDigis");
}

// -----------------------------------------------------------------------------
// destructor
SiStripDigiToRawModule::~SiStripDigiToRawModule() {
  if (verbosity_>1) cout << "[SiStripDigiToRawModule] "
			 << "destructing DigiToRaw module..." << endl;
  if ( digiToRaw_ ) delete digiToRaw_;
  if ( utility_ ) delete utility_; 
}

// -----------------------------------------------------------------------------
//
void SiStripDigiToRawModule::beginJob( const edm::EventSetup& iSetup ) {
  if (verbosity_>2) cout << "[SiStripDigiToRawModule::beginJob] "
			 << "creating utility object, connections map, DigiToRaw converter..." << endl;

  //@@ cannot presently retrieve connections map from SeventSetup!
  //   // retrieve cabling map (ESProduct) 
  //   ESHandle<SiStripConnection> connections;
  //   iSetup.get<TrackerConnectionRecord>().get( connections );
  //   cabling_.reset( connections.product() );

  // retrieve "dummy" connections map from utility object 
  utility_ = new SiStripUtility( iSetup );
  SiStripConnection connections;
  utility_->siStripConnection( connections );

  // create instance of DigiToRaw converter
  digiToRaw_ = new SiStripDigiToRaw( connections );
  // digiToRaw_->fedReadoutMode( fedReadoutMode_ );
  // digiToRaw_->fedReadoutPath( fedReadoutPath_ );

}

// -----------------------------------------------------------------------------
//
void SiStripDigiToRawModule::endJob() { 
  if (verbosity_>2) cout << "[SiStripDigiToRawModule::endJob] "
			 << "cuurently does nothing..." << endl;
}

// -----------------------------------------------------------------------------
// produces a FEDRawDataCollection
void SiStripDigiToRawModule::produce( edm::Event& iEvent, 
				      const edm::EventSetup& iSetup ) {
  if (verbosity_>2) cout << "[SiStripDigiToRawModule::produce] "
			 << "creates \"dummy\" StripDigiCollection as " << endl
			 << "input to DigiToRaw converter and writes "
			 << "\"FedRawDataCollection\" product to Event ..." << endl;
  
  event_++; // increment event counter
  if (verbosity_>0) cout << "[SiStripDigiToRawModule::produce] "
			 << "event number: " << event_ << endl;
  
  //   // retrieve collection of StripDigi's from Event
  //   edm::Handle<StripDigiCollection> input;
  //   e.getByLabel("StripDigiConverter", input);
  
  // retrieve "dummy" collection of StripDigi from utility object
  StripDigiCollection digis;
  utility_->stripDigiCollection( digis );

  // create product
  FEDRawDataCollection fed_buffers;

  // use DigiToRaw converter to fill FEDRawDataCollection
  digiToRaw_->createFedBuffers( digis, fed_buffers );

  // write FEDRawDataCollection to the Event
  // iEvent.put( &fed_buffers );

}

// -----------------------------------------------------------------------------
// define the class SiStripDigiToRawModule as a plug-in
//DEFINE_FWK_MODULE(SiStripDigiToRawModule)
