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
#include "FWCore/Utilities/interface/CRC16.h"

// GCT input data formats
#include "DataFormats/L1CaloTrigger/interface/L1CaloEmCand.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

// GCT output data formats
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

// Raw data collection
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

using std::cout;
using std::endl;
using std::vector;


GctDigiToRaw::GctDigiToRaw(const edm::ParameterSet& iConfig) :
  rctInputLabel_(iConfig.getParameter<edm::InputTag>("rctInputLabel")),
  gctInputLabel_(iConfig.getParameter<edm::InputTag>("gctInputLabel")),
  packRctEm_(iConfig.getUntrackedParameter<bool>("packRctEm", true)),
  packRctCalo_(iConfig.getUntrackedParameter<bool>("packRctCalo", true)),
  fedId_(iConfig.getParameter<int>("gctFedId")),
  verbose_(iConfig.getUntrackedParameter<bool>("verbose",false)),
  counter_(0),
  formatTranslator_()
{
  LogDebug("GCT") << "GctDigiToRaw will pack FED Id " << fedId_;

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
  formatTranslator_.setPackingBxId(bx);
  formatTranslator_.setPackingEventId(eventNumber);
 
  // The GCT and RCT input label strings
  const std::string gctInputLabelStr = gctInputLabel_.label();
  const std::string rctInputLabelStr = rctInputLabel_.label();
  
  // get GCT digis
  edm::Handle<L1GctEmCandCollection> isoEm;
  iEvent.getByLabel(gctInputLabelStr, "isoEm", isoEm);
  edm::Handle<L1GctEmCandCollection> nonIsoEm;
  iEvent.getByLabel(gctInputLabelStr, "nonIsoEm", nonIsoEm);
  edm::Handle<L1GctJetCandCollection> cenJets;
  iEvent.getByLabel(gctInputLabelStr, "cenJets", cenJets);
  edm::Handle<L1GctJetCandCollection> forJets;
  iEvent.getByLabel(gctInputLabelStr, "forJets", forJets);
  edm::Handle<L1GctJetCandCollection> tauJets;
  iEvent.getByLabel(gctInputLabelStr, "tauJets", tauJets);
  edm::Handle<L1GctEtTotalCollection> etTotal;
  iEvent.getByLabel(gctInputLabelStr, "", etTotal);
  edm::Handle<L1GctEtHadCollection> etHad;
  iEvent.getByLabel(gctInputLabelStr, "", etHad);
  edm::Handle<L1GctEtMissCollection> etMiss;
  iEvent.getByLabel(gctInputLabelStr, "", etMiss);
  edm::Handle<L1GctHFRingEtSumsCollection> hfRingSums;
  iEvent.getByLabel(gctInputLabelStr, "", hfRingSums);
  edm::Handle<L1GctHFBitCountsCollection> hfBitCounts;
  iEvent.getByLabel(gctInputLabelStr, "", hfBitCounts);
  edm::Handle<L1GctHtMissCollection> htMiss;
  iEvent.getByLabel(gctInputLabelStr, "", htMiss);
  edm::Handle<L1GctJetCountsCollection> jetCounts;
  iEvent.getByLabel(gctInputLabelStr, "", jetCounts);

  // get RCT EM Cand digi
  bool packRctEmThisEvent = packRctEm_;
  edm::Handle<L1CaloEmCollection> rctEm;
  if(packRctEmThisEvent)
  {
    iEvent.getByLabel(rctInputLabelStr, rctEm);
    if(rctEm.failedToGet())
    {
      packRctEmThisEvent = false;
      LogDebug("GCT") << "RCT EM Candidate packing requested, but failed to get them from event!";
    }
  }

  // get RCT Calo region digi
  bool packRctCaloThisEvent = packRctCalo_;
  edm::Handle<L1CaloRegionCollection> rctCalo;
  if(packRctCaloThisEvent)
  {
    iEvent.getByLabel(rctInputLabelStr, rctCalo);
    if(rctCalo.failedToGet())
    {
      packRctCaloThisEvent = false;
      LogDebug("GCT") << "RCT Calo Region packing requested, but failed to get them from event!";
    }
  }
  
  // create the raw data collection
  std::auto_ptr<FEDRawDataCollection> rawColl(new FEDRawDataCollection()); 
 
  // get the GCT buffer
  FEDRawData& fedRawData=rawColl->FEDData(fedId_);
 
  // set the size & make pointers to the header, beginning of payload, and footer.
  unsigned int rawSize = 88;  // MUST BE MULTIPLE OF 8! (slink packets are 64 bit, but using 8-bit data struct).
  if(packRctEmThisEvent) { rawSize += 232; }  // Space for RCT EM Cands.
  if(packRctCaloThisEvent) { rawSize += 800; }  // Space for RCT Calo Regions (plus a 32-bit word of padding to make divisible by 8)
  fedRawData.resize(rawSize);
  unsigned char * pHeader = fedRawData.data();  
  unsigned char * pPayload = pHeader + 16;  //  16 = 8 for slink header + 8 for Greg's versioning header.
  unsigned char * pFooter = pHeader + rawSize - 8;
 
  // Write CDF header (exactly as told by Marco Zanetti)
  FEDHeader fedHeader(pHeader);
  fedHeader.set(pHeader, 1, eventNumber, bx, fedId_);  // what should the bx_ID be?
 
  // Pack GCT jet output digis
  formatTranslator_.writeGctOutJetBlock(pPayload, 
                                        cenJets.product(),
                                        forJets.product(),
                                        tauJets.product(),
                                        hfRingSums.product(), 
                                        hfBitCounts.product(),
                                        htMiss.product());

  pPayload += 36; //advance payload pointer
  
  // Pack GCT EM and energy sums digis.
  formatTranslator_.writeGctOutEmAndEnergyBlock(pPayload,
                                                isoEm.product(), 
                                                nonIsoEm.product(),
                                                etTotal.product(), 
                                                etHad.product(), 
                                                etMiss.product());

  pPayload += 28; //advance payload pointer

  // Pack RCT EM Cands
  if(packRctEmThisEvent)
  {
    formatTranslator_.writeRctEmCandBlocks(pPayload, rctEm.product());
    pPayload+=232;  //advance payload pointer
  }

  // Pack RCT Calo Regions
  if(packRctCaloThisEvent)
  {
    formatTranslator_.writeAllRctCaloRegionBlock(pPayload, rctCalo.product());
  }
  
  // Write CDF footer (exactly as told by Marco Zanetti)
  FEDTrailer fedTrailer(pFooter);
  fedTrailer.set(pFooter, rawSize/8, evf::compute_crc(pHeader, rawSize), 0, 0);
 
  // Debug output.
  if (verbose_) { print(fedRawData); }
 
  // Put the collection in the event.
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
GctDigiToRaw::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
GctDigiToRaw::endJob() {
}

/// make this a plugin
DEFINE_FWK_MODULE(GctDigiToRaw);

