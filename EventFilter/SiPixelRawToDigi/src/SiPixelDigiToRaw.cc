
#include "EventFilter/SiPixelRawToDigi/interface/SiPixelDigiToRaw.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"


#include "EventFilter/SiPixelRawToDigi/interface/PixelDataFormatter.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDCabling.h"
using namespace std;

SiPixelDigiToRaw::SiPixelDigiToRaw( const edm::ParameterSet& pset ) :
  eventCounter_(0),
  fedCablingMap_(0),
  src_( pset.getParameter<edm::InputTag>( "src" ) )
{

  // Set some private data members
  //productLabel_ = pset.getParameter<std::string>("DigiProducer");

  // Define EDProduct type
  produces<FEDRawDataCollection>();

}

// -----------------------------------------------------------------------------
SiPixelDigiToRaw::~SiPixelDigiToRaw() {
}

// -----------------------------------------------------------------------------
void SiPixelDigiToRaw::beginJob(const edm::EventSetup& setup)
{
}

// -----------------------------------------------------------------------------
void SiPixelDigiToRaw::produce( edm::Event& ev,
                              const edm::EventSetup& es)
{
  eventCounter_++;
  edm::LogInfo("SiPixelDigiToRaw") << "[SiPixelDigiToRaw::produce] "
                        << "event number: "
                        << eventCounter_;


  PixelDataFormatter formatter;

  edm::Handle< edm::DetSetVector<PixelDigi> > digiCollection;
  ev.getByLabel( src_ , digiCollection);

  PixelDataFormatter::Digis digis;
  typedef vector< edm::DetSet<PixelDigi> >::const_iterator DI;
  
  for (DI di=digiCollection->begin(); di != digiCollection->end(); di++) {
    digis[ di->id] = di->data;
  }

  cout << " *** HERE1" << endl;
  edm::ESHandle<SiPixelFedCablingMap> map;
  es.get<SiPixelFedCablingMapRcd>().get( map );
  cout << map->version() << endl;
  cout << " *** HERE2" << endl;

  // create product (raw data)
  std::auto_ptr<FEDRawDataCollection> buffers( new FEDRawDataCollection );

  const vector<PixelFEDCabling> & cabling = map->cabling();

  typedef vector<PixelFEDCabling>::const_iterator FI;
  for (FI it = cabling.begin(); it != cabling.end(); it++) {
    LogDebug("SiPixelDigiToRaw")<<" PRODUCE DATA FOR FED_id: " << (*it).id();
    FEDRawData * rawData = formatter.formatData( (*it), digis);
    FEDRawData& fedRawData = buffers->FEDData( (*it).id() ); 
    fedRawData = *rawData;
    LogDebug("SiPixelDigiToRaw")<<"size of data in fedRawData: "<<fedRawData.size();
  }
  
  ev.put( buffers );
  
}

// -----------------------------------------------------------------------------

