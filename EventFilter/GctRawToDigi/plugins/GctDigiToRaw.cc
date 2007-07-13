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
   blockPacker_.setBcId(counter_ % 3564);
   blockPacker_.setEvId(counter_ % (0x1<<24));

   // get digis
   edm::Handle<L1GctEmCandCollection> isoEm;
   iEvent.getByLabel(gctInputLabel_.label(), "isoEm", isoEm);

   edm::Handle<L1GctEmCandCollection> nonIsoEm;
   iEvent.getByLabel(gctInputLabel_.label(), "nonIsoEm", nonIsoEm);

   // create the raw data collection
   std::auto_ptr<FEDRawDataCollection> rawColl(new FEDRawDataCollection()); 

   // get the GCT buffer
   FEDRawData& feddata=rawColl->FEDData(fedId_);

   // set the size & get pointer
   feddata.resize(48);
   unsigned char * d = feddata.data();

   // write CDF header
   blockPacker_.writeFedHeader(d, fedId_);
   d=d+8;

   // pack GCT EM output digis
   blockPacker_.writeGctEmBlock(d, isoEm.product(), nonIsoEm.product());      

   // write footer (is this necessary???)
   const unsigned char * s = feddata.data();
   blockPacker_.writeFedFooter(d, s);

   // debug output
   if (verbose_) print(feddata);

   // put the collection in the event
   iEvent.put(rawColl);

}


void GctDigiToRaw::print(FEDRawData& data) {

  const unsigned char * d = data.data();

  for (int i=0; i<data.size(); i=i+4) {
    uint32_t w = (uint32_t)d[i] + (uint32_t)(d[i+1]<<8) + (uint32_t)(d[i+2]<<16) + (uint32_t)(d[i+3]<<24);
    cout << std::hex << std::setw(4) << i/4 << " " << std::setw(8) << w << endl;
  }

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

