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
  : config_(conf), 
    cabling_(0), 
    hCPU(0), hDigi(0), theTimer(0)
{

  includeErrors = config_.getUntrackedParameter<bool>("IncludeErrors",false);
  checkOrder = config_.getUntrackedParameter<bool>("CheckPixelOrder",false);
  useCablingTree_ = config_.getUntrackedParameter<bool>("UseCablingTree",true);

  // Products
  produces< edm::DetSetVector<PixelDigi> >();
  if(includeErrors) produces< edm::DetSetVector<SiPixelRawDataError> >();

  // Timing
  bool timing = config_.getUntrackedParameter<bool>("Timing",false);
  if (timing) {
    theTimer = new R2DTimerObserver("**** MY TIMING REPORT ***");
    hCPU = new TH1D ("hCPU","hCPU",100,0.,0.050);
    hDigi = new TH1D("hDigi","hDigi",50,0.,15000.);
  }
}


// -----------------------------------------------------------------------------
SiPixelRawToDigi::~SiPixelRawToDigi() {
  edm::LogInfo("SiPixelRawToDigi")  << " HERE ** SiPixelRawToDigi destructor!";

  if(useCablingTree_) delete cabling_;

  if (theTimer) {
    TFile rootFile("analysis.root", "RECREATE", "my histograms");
    hCPU->Write();
    hDigi->Write();
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
  static std::vector<unsigned int> fedList;

// initialize cabling map or update if necessary
  static edm::ESWatcher<SiPixelFedCablingMapRcd> recordWatcher;
  if (recordWatcher.check( es )) {
    edm::ESHandle<SiPixelFedCablingMap> cablingMap;
    es.get<SiPixelFedCablingMapRcd>().get( cablingMap );
    fedList = cablingMap->fedIds();
    if(useCablingTree_ && cabling_) delete cabling_; 
    if (useCablingTree_) cabling_ = cablingMap->cablingTree(); 
    else cabling_ = cablingMap.product();
    LogDebug("map version:")<< cabling_->version();
  }

  edm::Handle<FEDRawDataCollection> buffers;
  static edm::InputTag label = config_.getUntrackedParameter<edm::InputTag>("InputLabel",edm::InputTag("source"));
  ev.getByLabel( label, buffers);

// create product (digis & errors)
  std::auto_ptr< edm::DetSetVector<PixelDigi> > collection( new edm::DetSetVector<PixelDigi> );
  std::auto_ptr< edm::DetSetVector<SiPixelRawDataError> > errorcollection( new edm::DetSetVector<SiPixelRawDataError> );
  static int ndigis = 0;
  static int nwords = 0;
  static uint32_t dummydetid = 0xffffffff;

  PixelDataFormatter formatter(cabling_);
  formatter.setErrorStatus(includeErrors, checkOrder);

  if (theTimer) theTimer->start();
  bool errorsInEvent = false;
  PixelDataFormatter::DetErrors nodeterrors;

  typedef std::vector<unsigned int>::const_iterator IF;
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
	if (errordetid==dummydetid) {           // errors given dummy detId must be sorted by Fed
	  nodeterrors.insert( nodeterrors.end(), errors[errordetid].begin(), errors[errordetid].end() );
	} else {
	  edm::DetSet<SiPixelRawDataError>& errorDetSet = errorcollection->find_or_insert(errordetid);
	  errorDetSet.data = is->second;
	}
      }
    }
  }

  if(includeErrors) {
    edm::DetSet<SiPixelRawDataError>& errorDetSet = errorcollection->find_or_insert(dummydetid);
    errorDetSet.data = nodeterrors;
  }
  if (errorsInEvent) LogDebug("SiPixelRawToDigi") << "Error words were stored in this event";

  if (theTimer) {
    theTimer->stop();
    LogDebug("SiPixelRawToDigi") << "TIMING IS: (real)" << theTimer->lastMeasurement().real() ;
    ndigis += formatter.nDigis();
    nwords += formatter.nWords();
    LogDebug("SiPixelRawToDigi") << " (Words/Digis) this ev: "
         <<formatter.nWords()<<"/"<<formatter.nDigis() << "--- all :"<<nwords<<"/"<<ndigis;
    hCPU->Fill( theTimer->lastMeasurement().real() ); 
    hDigi->Fill(formatter.nDigis());
  }

  //send digis and errors back to framework 
  ev.put( collection );
  if(includeErrors) ev.put( errorcollection );
}
