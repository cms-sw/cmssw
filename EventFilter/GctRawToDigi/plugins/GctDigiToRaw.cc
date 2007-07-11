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
  rctInputLabel_(iConfig.getParameter<edm::InputTag>("rctInputLabel")),
  gctInputLabel_(iConfig.getParameter<edm::InputTag>("gctInputLabel")),
  fedId_(iConfig.getParameter<int>("gctFedId")),
  verbose_(iConfig.getUntrackedParameter<bool>("verbose",false)),
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
   iEvent.getByLabel(gctInputLabel_.label(), "isoEm", isoEm);

   edm::Handle<L1GctEmCandCollection> nonIsoEm;
   iEvent.getByLabel(gctInputLabel_.label(), "nonIsoEm", nonIsoEm);

   // create the raw data collection
   std::auto_ptr<FEDRawDataCollection> rawColl(new FEDRawDataCollection()); 

   // get the GCT buffer
   FEDRawData& feddata=rawColl->FEDData(fedId_);
   unsigned char * d = feddata.data();

   // set the size
   feddata.resize(32);

   // write CDF header
   writeHeader(feddata);

   // pack GCT EM output digis
   blockPacker_.writeGctEmBlock(&d[8], isoEm);


   // write footer (is this necessary???)

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

