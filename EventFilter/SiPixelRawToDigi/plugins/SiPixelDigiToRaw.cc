#include "SiPixelDigiToRaw.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"


#include "EventFilter/SiPixelRawToDigi/interface/PixelDataFormatter.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDCabling.h"
using namespace std;

SiPixelDigiToRaw::SiPixelDigiToRaw( const edm::ParameterSet& pset ) :
  cablingTree_(0),
  config_(pset)
{

  // Define EDProduct type
  produces<FEDRawDataCollection>();

}

// -----------------------------------------------------------------------------
SiPixelDigiToRaw::~SiPixelDigiToRaw() {
   delete cablingTree_;
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
  static unsigned long eventCounter = 0;

  eventCounter++;
  edm::LogInfo("SiPixelDigiToRaw") << "[SiPixelDigiToRaw::produce] "
                        << "event number: "
                        << eventCounter;

  edm::Handle< edm::DetSetVector<PixelDigi> > digiCollection;
  //static string label = config_.getUntrackedParameter<string>("InputLabel","source");
  static edm::InputTag label = config_.getUntrackedParameter<edm::InputTag>("InputLabel",edm::InputTag("source"));
  ev.getByLabel( label, digiCollection);

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

  static edm::ESWatcher<SiPixelFedCablingMapRcd> recordWatcher;
  if (recordWatcher.check( es )) {
    edm::ESHandle<SiPixelFedCablingMap> cablingMap;
    es.get<SiPixelFedCablingMapRcd>().get( cablingMap );
    if (cablingTree_) delete cablingTree_; cablingTree_= cablingMap->cablingTree();
  }

  static bool debug = edm::MessageDrop::instance()->debugEnabled;
  if (debug) LogDebug("SiPixelDigiToRaw") << cablingTree_->version();
  
  PixelDataFormatter formatter(cablingTree_);

  // create product (raw data)
  std::auto_ptr<FEDRawDataCollection> buffers( new FEDRawDataCollection );

  const vector<const PixelFEDCabling *>  fedList = cablingTree_->fedList();

  typedef vector<const PixelFEDCabling *>::const_iterator FI;
  for (FI it = fedList.begin(); it != fedList.end(); it++) {
    LogDebug("SiPixelDigiToRaw")<<" PRODUCE DATA FOR FED_id: " << (**it).id();
    FEDRawData * rawData = formatter.formatData( ev.id().event(),(**it).id(), digis);
    FEDRawData& fedRawData = buffers->FEDData( (**it).id() ); 
    fedRawData = *rawData;
    LogDebug("SiPixelDigiToRaw")<<"size of data in fedRawData: "<<fedRawData.size();
    delete rawData;
  }
  allWordCounter += formatter.nWords();
  if (debug) LogDebug("SiPixelDigiToRaw") 
        << "Words/Digis this ev: "<<digiCounter<<"(fm:"<<formatter.nDigis()<<")/"
        <<formatter.nWords()
        <<"  all: "<< allDigiCounter <<"/"<<allWordCounter;

  
  ev.put( buffers );
  
}

// -----------------------------------------------------------------------------

