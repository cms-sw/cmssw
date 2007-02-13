#include "EventFilter/GctRawToDigi/src/GctRawToDigi.h"

// system
#include <vector>
#include <iostream>

// framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Raw data collection
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

// GCT raw data formats
//#include "EventFilter/GctRawToDigi/src/GctDaqRecord.h"
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

unsigned GctRawToDigi::MAX_EXCESS = 512;
unsigned GctRawToDigi::MAX_BLOCKS = 128;


GctRawToDigi::GctRawToDigi(const edm::ParameterSet& iConfig) :
  fedId_(iConfig.getUntrackedParameter<int>("GctFedId",745))
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
 
   // do anything here that needs to be done at destruction time
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

  std::vector<GctBlockHeader> bHdrs;


  // unpacking variables
  const unsigned char * data = d.data();
  unsigned dEnd = d.size()-16; // bytes in payload
  unsigned dPtr = 8; // data pointer
  bool lost = false;

  // read blocks
  for (unsigned nb=0; !lost && dPtr<dEnd && nb<MAX_BLOCKS; nb++) {

    // 1 read block header
    GctBlockHeader blockHead(&data[dPtr]);

    // 2 get block size (in 32 bit words)
    unsigned blockLen = converter_.blockLength(blockHead.id());

    // 3 if block recognised, convert it and store header
    if ( converter_.validBlock(blockHead.id()) ) {
      converter_.convertBlock(&data[dPtr], blockHead.id(), gctEm.get());
      bHdrs.push_back(blockHead);
      dPtr += 4*(blockLen+1); // 4 because blockLen is in 32-bit words, +1 for header
    }
    else {
      lost = true;
      edm::LogWarning("GCT") << "Unrecognised data block at byte " << dPtr << ". Bailing out" << endl;
      edm::LogWarning("GCT") << blockHead << endl;
    }
    
  }

  cout << "Found " << bHdrs.size() << " GCT internal headers" << endl;
  for (unsigned i=0; i<bHdrs.size(); i++) {
    cout << bHdrs[i]<< endl;
  }
  cout << "Read " << gctEm.get()->size() << " GCT EM candidates" << endl;

  

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

