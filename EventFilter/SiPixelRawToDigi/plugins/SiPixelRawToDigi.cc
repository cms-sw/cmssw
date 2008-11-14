#include "SiPixelRawToDigi.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"

#include "DataFormats/SiPixelRawData/interface/SiPixelRawDataError.h"


#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "EventFilter/SiPixelRawToDigi/interface/PixelDataFormatter.h"

#include "EventFilter/SiPixelRawToDigi/interface/R2DTimerObserver.h"

#include "TH1D.h"
#include "TFile.h"

using namespace std;

// -----------------------------------------------------------------------------
SiPixelRawToDigi::SiPixelRawToDigi( const edm::ParameterSet& conf ) 
  : config_(conf), cablingTree_(0),
    hCPU(0), hDigi(0), rootFile(0), theTimer(0)
{

  includeErrors = config_.getUntrackedParameter<bool>("IncludeErrors",false);
  checkOrder = config_.getUntrackedParameter<bool>("CheckPixelOrder",false);

  // Products
  produces< edm::DetSetVector<PixelDigi> >();
  if(includeErrors) produces< edm::DetSetVector<SiPixelRawDataError> >();

  // Timing
  bool timing = config_.getUntrackedParameter<bool>("Timing",false);
  if (timing) {
    theTimer = new R2DTimerObserver("**** MY TIMING REPORT ***");
    hCPU = new TH1D ("hCPU","hCPU",60,0.,0.030);
    hDigi = new TH1D("hDigi","hDigi",50,0.,15000.);
  }
}


// -----------------------------------------------------------------------------
SiPixelRawToDigi::~SiPixelRawToDigi() {
  edm::LogInfo("SiPixelRawToDigi")  << " HERE ** SiPixelRawToDigi destructor!";

  delete cablingTree_;

  if (theTimer) {
    rootFile = new TFile("analysis.root", "RECREATE", "my histograms");
    hCPU->Write();
    hDigi->Write();
    rootFile->Close();
    delete theTimer;
  }

}


// -----------------------------------------------------------------------------
void SiPixelRawToDigi::beginJob(const edm::EventSetup& c) 
{
}

// -----------------------------------------------------------------------------
void SiPixelRawToDigi::produce( edm::Event& ev,
                              const edm::EventSetup& es) 
{
  static bool debug = edm::MessageDrop::instance()->debugEnabled;
  static std::vector<int> fedList;

// initialize cabling map or update if necessary
  static edm::ESWatcher<SiPixelFedCablingMapRcd> recordWatcher;
  if (recordWatcher.check( es )) {
    edm::ESHandle<SiPixelFedCablingMap> cablingMap;
    es.get<SiPixelFedCablingMapRcd>().get( cablingMap );
    if(cablingTree_) delete cablingTree_; cablingTree_ = cablingMap->cablingTree();
    LogDebug("map version:")<< cablingTree_->version();
  
    typedef std::vector<const sipixelobjects::PixelFEDCabling *>::iterator FLI;
    std::vector<const sipixelobjects::PixelFEDCabling *> feds = cablingTree_->fedList();

    fedList.clear();
    for (FLI fedIds = feds.begin(); fedIds != feds.end(); fedIds++) {
      int fedId = (*fedIds)->id();
      fedList.push_back( fedId );
    }
  }

  edm::Handle<FEDRawDataCollection> buffers;
  static edm::InputTag label = config_.getUntrackedParameter<edm::InputTag>("InputLabel",edm::InputTag("source"));
  ev.getByLabel( label, buffers);

// create product (digis & errors)
  std::auto_ptr< edm::DetSetVector<PixelDigi> > collection( new edm::DetSetVector<PixelDigi> );
  std::auto_ptr< edm::DetSetVector<SiPixelRawDataError> > errorcollection( new edm::DetSetVector<SiPixelRawDataError> );
  static int ndigis = 0;
  static int nwords = 0;

  PixelDataFormatter formatter(cablingTree_);
  formatter.setErrorStatus(includeErrors, checkOrder);

  if (theTimer) theTimer->start();
  bool errorsInEvent = false;

  typedef std::vector<int>::iterator IF;
  for (IF aFed = fedList.begin(); aFed != fedList.end(); ++aFed) {
    int fedId = *aFed;
    if(debug) LogDebug("SiPixelRawToDigi")<< " PRODUCE DIGI FOR FED: " <<  fedId << endl;
    PixelDataFormatter::Digis digis;
    PixelDataFormatter::Errors errors;
     
    //get event data for this fed
    const FEDRawData& fedRawData = buffers->FEDData( fedId );

    //convert data to digi and strip off errors
    formatter.interpretRawData( errorsInEvent, fedId, fedRawData, digis, errors);

    //pack digi into collection
    typedef PixelDataFormatter::Digis::iterator ID;
    for (ID it = digis.begin(); it != digis.end(); it++) {
      uint32_t detid = it->first;
      edm::DetSet<PixelDigi>& detSet = collection->find_or_insert(detid);
      detSet.data = it->second;
    } 

    //pack errors into collection
    if(includeErrors) {
      typedef PixelDataFormatter::Errors::iterator IE;
      for (IE is = errors.begin(); is != errors.end(); is++) {
	uint32_t errordetid = is->first;
	edm::DetSet<SiPixelRawDataError>& errorDetSet = errorcollection->find_or_insert(errordetid);
	errorDetSet.data = is->second;
      }
    } 
  } 
  if (errorsInEvent) LogDebug("SiPixelRawToDigi") << "Error words were stored in this event";

  if (theTimer) {
    theTimer->stop();
    cout << "TIMING IS: (real)" << theTimer->lastMeasurement().real() << endl;
    ndigis += formatter.nDigis();
    nwords += formatter.nWords();
    cout << " (Words/Digis) this ev: "<<formatter.nWords()<<"/"<<formatter.nDigis()
         << "--- all :"<<nwords<<"/"<<ndigis<<endl;
    hCPU->Fill( theTimer->lastMeasurement().real() ); 
    hDigi->Fill(formatter.nDigis());
  }

  //send digis and errors back to framework 
  ev.put( collection );
  if(includeErrors) ev.put( errorcollection );
}
