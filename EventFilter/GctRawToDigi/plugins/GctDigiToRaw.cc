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
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/Provenance/interface/EventID.h"

// Header needed to computer CRCs
#include "EventFilter/Utilities/interface/Crc.h"

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
  counter_(0),
  blockPacker_()
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
  
  counter_++; // To "simulate" bunch crossings for now...
  unsigned int bx = counter_ % 3564;  // What's the proper way of doing this?
  EventNumber_t eventNumber = iEvent.id().event();
  
  // Supply bx and EvID to the packer so it can make internal capture block headers.
  blockPacker_.setBcId(bx);
  blockPacker_.setEvId(eventNumber);
 
  // get digis
  edm::Handle<L1GctEmCandCollection> isoEm;
  iEvent.getByLabel(gctInputLabel_.label(), "isoEm", isoEm);
  edm::Handle<L1GctEmCandCollection> nonIsoEm;
  iEvent.getByLabel(gctInputLabel_.label(), "nonIsoEm", nonIsoEm);
  edm::Handle<L1GctJetCandCollection> cenJets;
  iEvent.getByLabel(gctInputLabel_.label(), "cenJets", cenJets);
  edm::Handle<L1GctJetCandCollection> forJets;
  iEvent.getByLabel(gctInputLabel_.label(), "forJets", forJets);
  edm::Handle<L1GctJetCandCollection> tauJets;
  iEvent.getByLabel(gctInputLabel_.label(), "tauJets", tauJets);
  
  // create the raw data collection
  std::auto_ptr<FEDRawDataCollection> rawColl(new FEDRawDataCollection()); 
 
  // get the GCT buffer
  FEDRawData& fedRawData=rawColl->FEDData(fedId_);
 
  // set the size & make pointers to the header, beginning of payload, and footer.
  const unsigned int rawSize = 64;  // MUST BE MULTIPLE OF 8! (slink packets are 64 bit, but using 8-bit data struct).
  fedRawData.resize(rawSize);
  unsigned char * pHeader = fedRawData.data();  
  unsigned char * pPayload = pHeader + 8;
  unsigned char * pFooter = pHeader + rawSize - 8;
 
  // Write CDF header (exactly as told by Marco Zanetti)
  FEDHeader fedHeader(pHeader);
  fedHeader.set(pHeader, 1, eventNumber, bx, fedId_);  // what should the bx_ID be?
 
  // pack GCT Jet output digis
  blockPacker_.writeGctJetBlock(pPayload, cenJets.product(), forJets.product(), tauJets.product());
 
  // pack GCT EM output digis. Add offset of 28 to start of Slink payload to get EM block header position.
  blockPacker_.writeGctEmBlock(pPayload + 28, isoEm.product(), nonIsoEm.product());      
 
  // write footer (exactly as told by Marco Zanetti)
  FEDTrailer fedTrailer(pFooter);
  fedTrailer.set(pFooter, rawSize/8, evf::compute_crc(pHeader, rawSize), 0, 0);
 
  // debug output
  if (verbose_) print(fedRawData);
 
  // put the collection in the event
  iEvent.put(rawColl);
}


void GctDigiToRaw::print(FEDRawData& data) {

  const unsigned char * d = data.data();

  for (unsigned int i=0; i<data.size(); i=i+4) {
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

