#include "SiPixelDigiToRaw.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"

#include "EventFilter/SiPixelRawToDigi/interface/PixelDataFormatter.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDCabling.h"

#include "EventFilter/SiPixelRawToDigi/interface/R2DTimerObserver.h"

#include "TH1D.h"
#include "TFile.h"

using namespace std;

SiPixelDigiToRaw::SiPixelDigiToRaw( const edm::ParameterSet& pset ) :
  frameReverter_(nullptr),
  config_(pset),
  hCPU(0), hDigi(0), theTimer(0)
{

  tPixelDigi = consumes<edm::DetSetVector<PixelDigi> >(config_.getParameter<edm::InputTag>("InputLabel")); 
 // Define EDProduct type
  produces<FEDRawDataCollection>();

  // start the counters
  eventCounter = 0;
  allDigiCounter = 0;
  allWordCounter = 0;

  // Timing
  bool timing = config_.getUntrackedParameter<bool>("Timing",false);
  if (timing) {
    theTimer = new R2DTimerObserver("**** MY TIMING REPORT ***");
    hCPU = new TH1D ("hCPU","hCPU",100,0.,0.050);
    hDigi = new TH1D("hDigi","hDigi",50,0.,15000.);
  }
}

// -----------------------------------------------------------------------------
SiPixelDigiToRaw::~SiPixelDigiToRaw() {
  delete frameReverter_;

  if (theTimer) {
    TFile rootFile("analysis.root", "RECREATE", "my histograms");
    hCPU->Write();
    hDigi->Write();
    delete theTimer;
  }
}

// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
void SiPixelDigiToRaw::produce( edm::Event& ev,
                              const edm::EventSetup& es)
{
  using namespace sipixelobjects;
  eventCounter++;
  edm::LogInfo("SiPixelDigiToRaw") << "[SiPixelDigiToRaw::produce] "
                                   << "event number: " << eventCounter;

  edm::Handle< edm::DetSetVector<PixelDigi> > digiCollection;
  label = config_.getParameter<edm::InputTag>("InputLabel");
  ev.getByToken( tPixelDigi, digiCollection);

  PixelDataFormatter::RawData rawdata;
  PixelDataFormatter::Digis digis;
  typedef vector< edm::DetSet<PixelDigi> >::const_iterator DI;

  int digiCounter = 0; 
  for (DI di=digiCollection->begin(); di != digiCollection->end(); di++) {
    digiCounter += (di->data).size(); 
    digis[ di->id] = di->data;
  }
  allDigiCounter += digiCounter;

  if (recordWatcher.check( es )) {
    edm::ESHandle<SiPixelFedCablingMap> cablingMap;
    es.get<SiPixelFedCablingMapRcd>().get( cablingMap );
    fedIds = cablingMap->fedIds();
    cablingTree_= cablingMap->cablingTree();
    if (frameReverter_) delete frameReverter_; frameReverter_ = new SiPixelFrameReverter( es, cablingMap.product() );
  }

  debug = edm::MessageDrop::instance()->debugEnabled;
  if (debug) LogDebug("SiPixelDigiToRaw") << cablingTree_->version();

  PixelDataFormatter formatter(cablingTree_.get());
  formatter.passFrameReverter(frameReverter_);
  if (theTimer) theTimer->start();

  // create product (raw data)
  std::auto_ptr<FEDRawDataCollection> buffers( new FEDRawDataCollection );

  const vector<const PixelFEDCabling *>  fedList = cablingTree_->fedList();

  // convert data to raw
  formatter.formatRawData( ev.id().event(), rawdata, digis );

  // pack raw data into collection
  typedef vector<const PixelFEDCabling *>::const_iterator FI;
  for (FI it = fedList.begin(); it != fedList.end(); it++) {
    LogDebug("SiPixelDigiToRaw")<<" PRODUCE DATA FOR FED_id: " << (**it).id();
    FEDRawData& fedRawData = buffers->FEDData( (**it).id() );
    PixelDataFormatter::RawData::iterator fedbuffer = rawdata.find( (**it).id() );
    if( fedbuffer != rawdata.end() ) fedRawData = fedbuffer->second;
    LogDebug("SiPixelDigiToRaw")<<"size of data in fedRawData: "<<fedRawData.size();
  }
  allWordCounter += formatter.nWords();
  if (debug) LogDebug("SiPixelDigiToRaw") 

        << "Words/Digis this ev: "<<digiCounter<<"(fm:"<<formatter.nDigis()<<")/"
        <<formatter.nWords()
        <<"  all: "<< allDigiCounter <<"/"<<allWordCounter;

  if (theTimer) {
    theTimer->stop();
    LogDebug("SiPixelDigiToRaw") << "TIMING IS: (real)" << theTimer->lastMeasurement().real() ;
    LogDebug("SiPixelDigiToRaw") << " (Words/Digis) this ev: "
         <<formatter.nWords()<<"/"<<formatter.nDigis() << "--- all :"<<allWordCounter<<"/"<<allDigiCounter;
    hCPU->Fill( theTimer->lastMeasurement().real() ); 
    hDigi->Fill(formatter.nDigis());
  }
  
  ev.put( buffers );
  
}

// -----------------------------------------------------------------------------

