#include "EventFilter/GctRawToDigi/plugins/GctDigiToRaw.h"

// system
#include <vector>
#include <sstream>
#include <iostream>
#include <iomanip>

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


// Raw data collection
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

using std::cout;
using std::endl;
using std::vector;

//unsigned GctDigiToRaw::MAX_EXCESS = 512;
//unsigned GctDigiToRaw::MAX_BLOCKS = 128;


GctDigiToRaw::GctDigiToRaw(const edm::ParameterSet& iConfig) :
  inputLabel_(iConfig.getParameter<edm::InputTag>("inputLabel")),
  fedId_(iConfig.getParameter<int>("GctFedId")),
  verbose_(iConfig.getUntrackedParameter<bool>("Verbose",false)),
  counter_(0)
{

  edm::LogInfo("GCT") << "GctDigiToRaw will pack FED Id " << fedId_ << endl;

  //register the products
  produces<FEDRawDataCollection>();

}


GctDigiToRaw::~GctDigiToRaw()
{
 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
GctDigiToRaw::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   // counters
   counter_++;
   bx_ = (counter_ % 3564);
   lv1_ = counter_ % (0x1<<24);

   // get digis
   edm::Handle<L1GctEmCandCollection> isoEm;
   iEvent.getByLabel(inputLabel_.label(), "isoEm", isoEm);

   edm::Handle<L1GctEmCandCollection> nonIsoEm;
   iEvent.getByLabel(inputLabel_.label(), "nonIsoEm", nonIsoEm);

   // set digi collections in converter
   converter_.setIsoEmCollection(const_cast<L1GctEmCandCollection*>(isoEm.product()));
   converter_.setNonIsoEmCollection(const_cast<L1GctEmCandCollection*>(nonIsoEm.product()));

   // create the collection
   std::auto_ptr<FEDRawDataCollection> rawColl(new FEDRawDataCollection()); 
   // retrieve the target buffer
   FEDRawData& feddata=rawColl->FEDData(fedId_);
   unsigned char * d = feddata.data();

   // set the size
   feddata.resize(32);

   // write header
   writeHeader(feddata);

   // pack GCT EM output digis
   converter_.writeBlock(&d[8], 0x68);

   // write footer (do i have to do this?!?)


   // debug print out
   for (int i=0; i<feddata.size()/4; i++) {
     unsigned ptr = i*4;
     unsigned * w32 = reinterpret_cast<unsigned*>(&d[ptr]);
     cout << i << " " << std::hex << std::setw(8) << (*w32) << endl;
   }

   // put the collection in the event
   iEvent.put(rawColl);

}


// write Common Data Format header (nicked from EcalDigiToRaw)
void GctDigiToRaw::writeHeader(FEDRawData& data) {

  typedef long long Word64;

  // Allocate space for header+trailer+payload
  data.resize(8);
  
  // Standard FEVT header
  unsigned long long hdr;
  hdr = 0x18
    + ((fedId_ & 0xFFF)<<8)
    + ((Word64)((Word64)bx_ & 0xFFF)<<20)
    + ((Word64)((Word64)lv1_ & 0xFFFFFF)<<32)
    + ((Word64)((Word64)0x51<<56));

  unsigned char * pData = data.data();
  Word64* pw = reinterpret_cast<Word64*>(const_cast<unsigned char*>(pData));
  *pw = hdr;
  
}

// ------------ method called once each job just before starting event loop  ------------
void 
GctDigiToRaw::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
GctDigiToRaw::endJob() {
}



/// make this a plugin
DEFINE_FWK_MODULE(GctDigiToRaw);

