#include "EventFilter/GctRawToDigi/src/GctRawToDigi.h"

// system
#include <vector>

// framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Raw data collection
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

// GCT raw data formats
#include "EventFilter/GctRawToDigi/src/GctDaqRecord.h"
#include "EventFilter/GctRawToDigi/src/GctBlock.h"

// GCT input data formats
#include "DataFormats/L1CaloTrigger/interface/L1CaloEmCand.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

// GCT output data formats
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCounts.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"


using std::cout;
using std::endl;
using std::vector;


GctRawToDigi::GctRawToDigi(const edm::ParameterSet& iConfig) :
  fedId_(iConfig.getUntrackedParameter<int>("GctFedId",999))
{

  edm::LogInfo("GCT") << "GctRawToDigi will unpack FED Id " << fedId_ << endl;

  //register the products
  produces<L1CaloEmCollection>();
  produces<L1CaloRegionCollection>();
  produces<L1GctEmCandCollection>();
  produces<L1GctJetCandCollection>();
  produces<L1GctEtTotal>();
  produces<L1GctEtHad>();
  produces<L1GctEtMiss>();
  produces<L1GctJetCounts>();

}


GctRawToDigi::~GctRawToDigi()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
GctRawToDigi::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   // get raw data collection (by type?!?)
   edm::Handle<FEDRawDataCollection> feds;
   iEvent.getByType(feds);
   const FEDRawData& gctRcd = feds->FEDData(fedId_);
   
   unpack(gctRcd, iEvent);


/* this is an EventSetup example
   //Read SetupData from the SetupRecord in the EventSetup
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
*/
}

void GctRawToDigi::unpack(const FEDRawData& d, edm::Event& e) {

  cout << "Unpacking an event" << endl;

  // do a simple check of the raw data
  if (d.size()<16) {
      edm::LogWarning("Invalid Data") << "Empty/invalid GCT raw data, size = " << d.size();
      return;
  }


  // make collections for storing data
  std::auto_ptr<L1CaloEmCollection> rctEm(new L1CaloEmCollection()); 
  std::auto_ptr<L1CaloRegionCollection> rctRgn(new L1CaloRegionCollection()); 
  
  std::auto_ptr<L1GctEmCandCollection> gctEm(new L1GctEmCandCollection()); 
  std::auto_ptr<L1GctJetCandCollection> gctRgn(new L1GctJetCandCollection()); 
    
  // unpack process :
  // 1. read internal header
  // 2. read following nSamples, and
  //    2a. if blockId is known, create relevant objects
  //    2b. otherwise, create block
  // 3. move to next internal header
  
  GctDaqRecord rcd(d.data(), d.size());
  
  cout << rcd << endl;

}


// ------------ method called once each job just before starting event loop  ------------
void 
GctRawToDigi::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
GctRawToDigi::endJob() {
}

