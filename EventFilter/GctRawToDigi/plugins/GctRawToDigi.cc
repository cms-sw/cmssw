#include "EventFilter/GctRawToDigi/plugins/GctRawToDigi.h"

// system
#include <vector>
#include <sstream>
#include <iostream>

// framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"


// Raw data collection
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

// GCT raw data formats
#include "EventFilter/GctRawToDigi/src/GctBlockHeader.h"
//#include "EventFilter/GctRawToDigi/interface/L1GctInternalObject.h"

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
  inputLabel_(iConfig.getParameter<edm::InputTag>("inputLabel")),
  fedId_(iConfig.getParameter<int>("gctFedId")),
  verbose_(iConfig.getUntrackedParameter<bool>("verbose",false)),
  doEm_(iConfig.getUntrackedParameter<bool>("unpackEm",true)),
  doJets_(iConfig.getUntrackedParameter<bool>("unpackJets",true)),
  doEtSums_(iConfig.getUntrackedParameter<bool>("unpackEtSums",true)),
  doInternEm_(iConfig.getUntrackedParameter<bool>("unpackInternEm",true)),
  doFibres_(iConfig.getUntrackedParameter<bool>("unpackFibres",true))
{

  edm::LogInfo("GCT") << "GctRawToDigi will unpack FED Id " << fedId_ << endl;

  //register the products
  produces<L1CaloEmCollection>();
  produces<L1CaloRegionCollection>();
  produces<L1GctEmCandCollection>("isoEm");
  produces<L1GctEmCandCollection>("nonIsoEm");
  produces<L1GctInternEmCandCollection>();
  produces<L1GctJetCandCollection>("cenJets");
  produces<L1GctJetCandCollection>("forJets");
  produces<L1GctJetCandCollection>("tauJets");
  produces<L1GctEtTotal>();
  produces<L1GctEtHad>();
  produces<L1GctEtMiss>();
  produces<L1GctJetCounts>();
  produces<L1GctFibreCollection>();

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

   // get raw data collection
   edm::Handle<FEDRawDataCollection> feds;
   iEvent.getByLabel(inputLabel_, feds);
   const FEDRawData& gctRcd = feds->FEDData(fedId_);
   
   edm::LogInfo("GCT") << "Upacking FEDRawData of size " << std::dec << gctRcd.size() << std::endl;

  // do a simple check of the raw data
  if (gctRcd.size()<16) {
      edm::LogWarning("Invalid Data") << "Empty/invalid GCT raw data, size = " << gctRcd.size();
      return;
  }
  else {
    unpack(gctRcd, iEvent);
  }

}


void GctRawToDigi::unpack(const FEDRawData& d, edm::Event& e) {

  // make collections for storing data

  // Block headers
  std::vector<GctBlockHeader> bHdrs;

  // GCT input data
  std::auto_ptr<L1CaloEmCollection> rctEm( new L1CaloEmCollection() ); 
  std::auto_ptr<L1CaloRegionCollection> rctRgn( new L1CaloRegionCollection() ); 
  
  // GCT intermediate data
  std::auto_ptr<L1GctInternEmCandCollection> gctInternEm( new L1GctInternEmCandCollection() ); 

  // GCT output data
  std::auto_ptr<L1GctEmCandCollection> gctIsoEm( new L1GctEmCandCollection() ); 
  std::auto_ptr<L1GctEmCandCollection> gctNonIsoEm( new L1GctEmCandCollection() ); 
  std::auto_ptr<L1GctJetCandCollection> gctCenJets( new L1GctJetCandCollection() ); 
  std::auto_ptr<L1GctJetCandCollection> gctForJets( new L1GctJetCandCollection() ); 
  std::auto_ptr<L1GctJetCandCollection> gctTauJets( new L1GctJetCandCollection() );
  
  std::auto_ptr<L1GctEtTotal> etTotResult( new L1GctEtTotal() );
  std::auto_ptr<L1GctEtHad> etHadResult( new L1GctEtHad() );
  std::auto_ptr<L1GctEtMiss> etMissResult( new L1GctEtMiss() );

  std::auto_ptr<L1GctFibreCollection> gctFibres( new L1GctFibreCollection() );


  // setup blockUnpackerer
  blockUnpacker_.setRctEmCollection( rctEm.get() );
  blockUnpacker_.setIsoEmCollection( gctIsoEm.get() );
  blockUnpacker_.setNonIsoEmCollection( gctNonIsoEm.get() );
  blockUnpacker_.setInternEmCollection( gctInternEm.get() );
  blockUnpacker_.setFibreCollection( gctFibres.get() );

  // unpacking variables
  const unsigned char * data = d.data();
  unsigned dEnd = d.size()-16; // bytes in payload
  unsigned dPtr = 8; // data pointer
  bool lost = false;

  // read blocks
  for (unsigned nb=0; !lost && dPtr<dEnd && nb<MAX_BLOCKS; nb++) {

    // read block header
    GctBlockHeader blockHead(&data[dPtr]);

    // unpack the block
    blockUnpacker_.convertBlock(&data[dPtr+4], blockHead);

    // store the header
    bHdrs.push_back(blockHead);
    
    // advance pointer
    unsigned blockLen = blockHead.length();
    unsigned nSamples = blockHead.nSamples();
    dPtr += 4*(blockLen*nSamples+1); // *4 because blockLen is in 32-bit words, +1 for header
    
  }
  
  // dump summary in verbose mode
  if (verbose_) {
    std::ostringstream os;
    os << "Found " << bHdrs.size() << " GCT internal headers" << endl;
    for (unsigned i=0; i<bHdrs.size(); i++) {
      os << bHdrs[i]<< endl;
    }
    os << "Read " << rctEm.get()->size() << " RCT EM candidates" << endl;
    os << "Read " << gctIsoEm.get()->size() << " GCT iso EM candidates" << endl;
    os << "Read " << gctNonIsoEm.get()->size() << " GCT non-iso EM candidates" << endl;
    os << "Read " << gctInternEm.get()->size() << " GCT intermediate EM candidates" << endl;
    
    edm::LogVerbatim("GCT") << os.str();
  }


  // put data into the event
  if (doEm_) {
    e.put(rctEm);
    e.put(gctIsoEm, "isoEm");
    e.put(gctNonIsoEm, "nonIsoEm");
  }
  if (doJets_) {
    e.put(rctRgn);
    e.put(gctCenJets,"cenJets");
    e.put(gctForJets,"forJets");
    e.put(gctTauJets,"tauJets");
  }
  if (doEtSums_) {
  }
  if (doInternEm_) {
    e.put(gctInternEm);
  }
  if (doFibres_) {
    e.put(gctFibres);
  }

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



/// make this a plugin
DEFINE_FWK_MODULE(GctRawToDigi);

