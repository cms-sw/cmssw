#include "SiPixelDigiToRaw.h"
#include "DataFormats/Common/interface/Handle.h"
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
  config_(pset)
//  src_( pset.getParameter<edm::InputTag>( "src" ) )
{

  // Define EDProduct type
  string label = pset.getUntrackedParameter<string>("ProductLabel","sourceXXL");
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
  using namespace sipixelobjects;

  eventCounter_++;
  edm::LogInfo("SiPixelDigiToRaw") << "[SiPixelDigiToRaw::produce] "
                        << "event number: "
                        << eventCounter_;
  cout << " -- event:" << eventCounter_ << endl;

  edm::Handle< edm::DetSetVector<PixelDigi> > digiCollection;
  static string label = config_.getUntrackedParameter<string>("InputLabel","source");
  static string instance = config_.getUntrackedParameter<string>("InputInstance","");
  ev.getByLabel( label, instance, digiCollection);

  PixelDataFormatter::Digis digis;
  typedef vector< edm::DetSet<PixelDigi> >::const_iterator DI;

  static int allDigiCounter = 0;  
  static int allWordCounter = 0;
         int digiCounter = 0; 
  for (DI di=digiCollection->begin(); di != digiCollection->end(); di++) {
    digiCounter += (di->data).size(); 
    digis[ di->id] = di->data;
//    digis.push_back(*di);
  }
  allDigiCounter += digiCounter;

  edm::ESHandle<SiPixelFedCablingMap> map;
  es.get<SiPixelFedCablingMapRcd>().get( map );
  cout << map->version() << endl;
  
  PixelDataFormatter formatter(map.product());

  // create product (raw data)
  std::auto_ptr<FEDRawDataCollection> buffers( new FEDRawDataCollection );

  const vector<const PixelFEDCabling *>  cabling = map->fedList();

  typedef vector<const PixelFEDCabling *>::const_iterator FI;
  for (FI it = cabling.begin(); it != cabling.end(); it++) {
    LogDebug("SiPixelDigiToRaw")<<" PRODUCE DATA FOR FED_id: " << (**it).id();
    FEDRawData * rawData = formatter.formatData( (**it).id(), digis);
    FEDRawData& fedRawData = buffers->FEDData( (**it).id() ); 
    fedRawData = *rawData;
    LogDebug("SiPixelDigiToRaw")<<"size of data in fedRawData: "<<fedRawData.size();
  }
  allWordCounter += formatter.nWords();
  cout << "Words/Digis this ev: "<<digiCounter<<"(fm:"<<formatter.nDigis()<<")/"
        <<formatter.nWords()
       <<"  all: "<< allDigiCounter <<"/"<<allWordCounter<<endl;
  
  ev.put( buffers );
  
}

// -----------------------------------------------------------------------------

