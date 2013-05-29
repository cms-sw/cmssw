// Include parameter driven interface to SiPixelQuality for study purposes
// exclude ROC(raw) based on bad ROC list in SiPixelQuality
// enabled by: process.siPixelDigis.UseQualityInfo = True
// 20-10-2010 Andrew York (Tennessee)

#include "SiPixelRawToDigi.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"

#include "DataFormats/SiPixelRawData/interface/SiPixelRawDataError.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "EventFilter/SiPixelRawToDigi/interface/PixelDataFormatter.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"

#include "EventFilter/SiPixelRawToDigi/interface/R2DTimerObserver.h"

#include "TH1D.h"
#include "TFile.h"

using namespace std;

// -----------------------------------------------------------------------------
SiPixelRawToDigi::SiPixelRawToDigi( const edm::ParameterSet& conf ) 
  : config_(conf), 
    cabling_(0), 
    badPixelInfo_(0), 
    hCPU(0), hDigi(0), theTimer(0)
{

  includeErrors = config_.getParameter<bool>("IncludeErrors");
  useQuality = config_.getParameter<bool>("UseQualityInfo");
  useCablingTree_ = config_.getUntrackedParameter<bool>("UseCablingTree",true);
  if (config_.exists("ErrorList")) {
    errorList = config_.getParameter<std::vector<int> > ("ErrorList");
  }

  //start counters
  ndigis = 0;
  nwords = 0;

  // Products
  produces< edm::DetSetVector<PixelDigi> >();
  if(includeErrors){
    produces< edm::DetSetVector<SiPixelRawDataError> >();
    produces<DetIdCollection>();
  }

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


// -----------------------------------------------------------------------------
void SiPixelRawToDigi::produce( edm::Event& ev,
                              const edm::EventSetup& es) 
{
  const uint32_t dummydetid = 0xffffffff;
  debug = edm::MessageDrop::instance()->debugEnabled;

// initialize cabling map or update if necessary
  if (recordWatcher.check( es )) {
    // cabling map, which maps online address (fed->link->ROC->local pixel) to offline (DetId->global pixel)
    if (useCablingTree_) {
      delete cabling_;
      // we are going to make our own copy so safe to let the map be deleted early
      edm::ESTransientHandle<SiPixelFedCablingMap> cablingMap;
      es.get<SiPixelFedCablingMapRcd>().get( cablingMap );
      fedIds   = cablingMap->fedIds();
      cabling_ = cablingMap->cablingTree();
    } else {
      // we are going to hold the pointer so we need the map to stick around
      edm::ESHandle<SiPixelFedCablingMap> cablingMap;
      es.get<SiPixelFedCablingMapRcd>().get( cablingMap );
      fedIds   = cablingMap->fedIds();
      cabling_ = cablingMap.product();
    }
    LogDebug("map version:")<< cabling_->version();
  }
// initialize quality record or update if necessary
  if (qualityWatcher.check( es )&&useQuality) {
    // quality info for dead pixel modules or ROCs
    edm::ESHandle<SiPixelQuality> qualityInfo;
    es.get<SiPixelQualityRcd>().get( qualityInfo );
    badPixelInfo_ = qualityInfo.product();
    if (!badPixelInfo_) {
      edm::LogError("**SiPixelRawToDigi**")<<" Configured to use SiPixelQuality, but SiPixelQuality not present"<<endl;
    }
  }

  edm::Handle<FEDRawDataCollection> buffers;
  label = config_.getParameter<edm::InputTag>("InputLabel");
  ev.getByLabel( label, buffers);

// create product (digis & errors)
  std::auto_ptr< edm::DetSetVector<PixelDigi> > collection( new edm::DetSetVector<PixelDigi> );
  std::auto_ptr< edm::DetSetVector<SiPixelRawDataError> > errorcollection( new edm::DetSetVector<SiPixelRawDataError> );
  std::auto_ptr< DetIdCollection > error_detidcollection(new DetIdCollection());

  PixelDataFormatter formatter(cabling_);
  formatter.setErrorStatus(includeErrors);
  if (useQuality) formatter.setQualityStatus(useQuality, badPixelInfo_);

  if (theTimer) theTimer->start();
  bool errorsInEvent = false;
  PixelDataFormatter::DetErrors nodeterrors;

  typedef std::vector<unsigned int>::const_iterator IF;
  for (IF aFed = fedIds.begin(); aFed != fedIds.end(); ++aFed) {
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
	  // Fill detid of the detectors where there is error AND the error number is listed
	  // in the configurable error list in the job option cfi.
	  // Code needs to be here, because there can be a set of errors for each 
	  // entry in the for loop over PixelDataFormatter::Errors
	  if(!errorList.empty()){
	    DetId errorDetId(errordetid);
	    edm::DetSet<SiPixelRawDataError>::const_iterator itPixelError=errorDetSet.begin();
	    for(; itPixelError!=errorDetSet.end(); ++itPixelError){
	      std::vector<int>::iterator it_find = find(errorList.begin(), errorList.end(), itPixelError->getType());
	      if(it_find != errorList.end()){
		error_detidcollection->push_back(errordetid);
	      }
	    }
	  }
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
  if(includeErrors){
    ev.put( errorcollection );
    ev.put( error_detidcollection );
  }
}
