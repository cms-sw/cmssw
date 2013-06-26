// -*- C++ -*-
//
// Package:    HLTDummyCollections
// Class:      HLTDummyCollections
// 
/**\class HLTDummyCollections HLTDummyCollections.cc HLTrigger/HLTDummyCollections/src/HLTDummyCollections.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
 */
//
// Original Author:  Emmanuelle Perez
//         Created:  Tue May 19 09:54:19 CEST 2009
// $Id: HLTDummyCollections.cc,v 1.4 2010/07/30 04:55:49 wmtan Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// -- Ecal
#include "EventFilter/EcalRawToDigi/plugins/EcalRawToRecHitFacility.h"
// -- Hcal
#include "EventFilter/HcalRawToDigi/plugins/HcalRawToDigi.h"
// -- Ecal Preshower
#include "EventFilter/ESRawToDigi/interface/ESRawToDigi.h"
// -- Muons DT
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>
#include <DataFormats/DTDigi/interface/DTLocalTriggerCollection.h>
// -- Muons CSC
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCRPCDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
// -- SiPixels
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/Common/interface/DetSetVector.h"
// -- SiStrips
#include "DataFormats/Common/interface/LazyGetter.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
// --- GCT
#include "EventFilter/GctRawToDigi/plugins/GctRawToDigi.h"

// -- ObjectMap
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"


//
// class decleration
//

class HLTDummyCollections : public edm::EDProducer {
  public:
    explicit HLTDummyCollections(const edm::ParameterSet&);
    ~HLTDummyCollections();

  private:
    virtual void produce(edm::Event&, const edm::EventSetup&);

    // ----------member data ---------------------------

    std::string action_;
    bool doEcal_ ;
    bool doHcal_; 
    bool unpackZDC_ ;
    bool doEcalPreshower_ ;
    std::string ESdigiCollection_;
    bool doMuonDTDigis_ ;
    bool doMuonCSCDigis_ ;
    bool doSiPixelDigis_;
    bool doSiStrip_ ;
    bool doGCT_ ;
    bool doObjectMap_ ;

};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
HLTDummyCollections::HLTDummyCollections(const edm::ParameterSet& iConfig)
{
  action_           = iConfig.getParameter<std::string>("action");
  unpackZDC_        = iConfig.getParameter<bool>("UnpackZDC");
  ESdigiCollection_ = iConfig.getParameter<std::string>("ESdigiCollection");

  doEcal_           = ( action_ == "doEcal");
  doHcal_           = ( action_ == "doHcal");
  doEcalPreshower_  = ( action_ == "doEcalPreshower");
  doMuonDTDigis_    = ( action_ == "doMuonDT");
  doMuonCSCDigis_   = ( action_ == "doMuonCSC");
  doSiPixelDigis_   = ( action_ == "doSiPixel");
  doSiStrip_        = ( action_ == "doSiStrip");
  doObjectMap_      = ( action_ == "doObjectMap");
  doGCT_	    = ( action_ == "doGCT");

  if (doEcal_) {
    // ECAL unpacking :
    produces< edm::LazyGetter<EcalRecHit> >();
  }

  if (doHcal_) {
    // HCAL unpacking
    produces<HBHEDigiCollection>();
    produces<HFDigiCollection>();
    produces<HODigiCollection>();
    produces<HcalTrigPrimDigiCollection>();
    produces<HOTrigPrimDigiCollection>();
    if (unpackZDC_) {
      produces<ZDCDigiCollection>();
    }
  }

  if (doEcalPreshower_) {
    produces<ESDigiCollection>();
  }

  if (doMuonDTDigis_) {
    produces<DTDigiCollection>();
    produces<DTLocalTriggerCollection>();
  }

  if (doMuonCSCDigis_) {
    produces<CSCWireDigiCollection>("MuonCSCWireDigi");
    produces<CSCStripDigiCollection>("MuonCSCStripDigi");
    produces<CSCALCTDigiCollection>("MuonCSCALCTDigi");
    produces<CSCCLCTDigiCollection>("MuonCSCCLCTDigi");
    produces<CSCComparatorDigiCollection>("MuonCSCComparatorDigi");
    produces<CSCRPCDigiCollection>("MuonCSCRPCDigi");
    produces<CSCCorrelatedLCTDigiCollection>("MuonCSCCorrelatedLCTDigi");
  }


  if (doSiPixelDigis_) {
    produces< edm::DetSetVector<PixelDigi> >();
  }

  if (doSiStrip_) {
    produces< edm::LazyGetter<SiStripCluster> >();
  }

  if (doGCT_) {
    // GCT output collections
    produces<L1GctEmCandCollection>("isoEm");
    produces<L1GctEmCandCollection>("nonIsoEm");
    produces<L1GctJetCandCollection>("cenJets");
    produces<L1GctJetCandCollection>("forJets");
    produces<L1GctJetCandCollection>("tauJets");
    produces<L1GctHFBitCountsCollection>();
    produces<L1GctHFRingEtSumsCollection>();
    produces<L1GctEtTotalCollection>();
    produces<L1GctEtHadCollection>();
    produces<L1GctEtMissCollection>();
    produces<L1GctHtMissCollection>();
    produces<L1GctJetCountsCollection>();  // Deprecated (empty collection still needed by GT)
  }

  if (doObjectMap_) {
    produces<L1GlobalTriggerObjectMapRecord>();
  }
}

HLTDummyCollections::~HLTDummyCollections()
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
HLTDummyCollections::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  if (doEcal_) {
    std::auto_ptr< edm::LazyGetter<EcalRecHit> > Ecalcollection( new edm::LazyGetter<EcalRecHit> );
    iEvent.put(Ecalcollection);
  }

  if (doHcal_) {
    std::auto_ptr<HBHEDigiCollection> hbhe_prod(new HBHEDigiCollection()); 
    std::auto_ptr<HFDigiCollection> hf_prod(new HFDigiCollection());
    std::auto_ptr<HODigiCollection> ho_prod(new HODigiCollection());
    std::auto_ptr<HcalTrigPrimDigiCollection> htp_prod(new HcalTrigPrimDigiCollection());  
    std::auto_ptr<HOTrigPrimDigiCollection> hotp_prod(new HOTrigPrimDigiCollection());  
    iEvent.put(hbhe_prod);
    iEvent.put(hf_prod);
    iEvent.put(ho_prod);
    iEvent.put(htp_prod);
    iEvent.put(hotp_prod);
    if (unpackZDC_) {
      std::auto_ptr<ZDCDigiCollection> zdcprod(new ZDCDigiCollection());
      iEvent.put(zdcprod);
    }
  }

  if (doEcalPreshower_) {
    std::auto_ptr<ESDigiCollection> productDigis(new ESDigiCollection);  
    iEvent.put(productDigis, ESdigiCollection_);
  }

  if (doMuonDTDigis_) {
    std::auto_ptr<DTDigiCollection> detectorProduct(new DTDigiCollection);
    std::auto_ptr<DTLocalTriggerCollection> triggerProduct(new DTLocalTriggerCollection);
    iEvent.put(detectorProduct);
    iEvent.put(triggerProduct);
  }

  if (doMuonCSCDigis_) {
    std::auto_ptr<CSCWireDigiCollection> wireProduct(new CSCWireDigiCollection);
    std::auto_ptr<CSCStripDigiCollection> stripProduct(new CSCStripDigiCollection);
    std::auto_ptr<CSCALCTDigiCollection> alctProduct(new CSCALCTDigiCollection);
    std::auto_ptr<CSCCLCTDigiCollection> clctProduct(new CSCCLCTDigiCollection);
    std::auto_ptr<CSCComparatorDigiCollection> comparatorProduct(new CSCComparatorDigiCollection);
    std::auto_ptr<CSCRPCDigiCollection> rpcProduct(new CSCRPCDigiCollection);
    std::auto_ptr<CSCCorrelatedLCTDigiCollection> corrlctProduct(new CSCCorrelatedLCTDigiCollection);

    iEvent.put(wireProduct,"MuonCSCWireDigi");
    iEvent.put(stripProduct,"MuonCSCStripDigi");
    iEvent.put(alctProduct,"MuonCSCALCTDigi");
    iEvent.put(clctProduct,"MuonCSCCLCTDigi");
    iEvent.put(comparatorProduct,"MuonCSCComparatorDigi");
    iEvent.put(rpcProduct,"MuonCSCRPCDigi");
    iEvent.put(corrlctProduct,"MuonCSCCorrelatedLCTDigi");
  }

  if (doSiPixelDigis_) {
    std::auto_ptr< edm::DetSetVector<PixelDigi> > SiPicollection( new edm::DetSetVector<PixelDigi> );
    iEvent.put(SiPicollection);
  }

  if (doSiStrip_) {
    std::auto_ptr< edm::LazyGetter<SiStripCluster> > SiStripcollection( new edm::LazyGetter<SiStripCluster> );
    iEvent.put(SiStripcollection);
  }

  if (doGCT_) {
    std::auto_ptr<L1GctEmCandCollection> m_gctIsoEm( new L1GctEmCandCollection) ;
    std::auto_ptr<L1GctEmCandCollection> m_gctNonIsoEm(new L1GctEmCandCollection);
    std::auto_ptr<L1GctJetCandCollection> m_gctCenJets(new L1GctJetCandCollection);
    std::auto_ptr<L1GctJetCandCollection> m_gctForJets(new L1GctJetCandCollection);
    std::auto_ptr<L1GctJetCandCollection> m_gctTauJets(new L1GctJetCandCollection);
    std::auto_ptr<L1GctHFBitCountsCollection> m_gctHfBitCounts(new L1GctHFBitCountsCollection);
    std::auto_ptr<L1GctHFRingEtSumsCollection> m_gctHfRingEtSums(new L1GctHFRingEtSumsCollection);
    std::auto_ptr<L1GctEtTotalCollection> m_gctEtTot(new L1GctEtTotalCollection);
    std::auto_ptr<L1GctEtHadCollection> m_gctEtHad(new L1GctEtHadCollection);
    std::auto_ptr<L1GctEtMissCollection> m_gctEtMiss(new L1GctEtMissCollection);
    std::auto_ptr<L1GctHtMissCollection> m_gctHtMiss(new L1GctHtMissCollection);
    std::auto_ptr<L1GctJetCountsCollection> m_gctJetCounts(new L1GctJetCountsCollection);  // DEPRECATED

    iEvent.put(m_gctIsoEm, "isoEm");
    iEvent.put(m_gctNonIsoEm, "nonIsoEm");
    iEvent.put(m_gctCenJets,"cenJets");
    iEvent.put(m_gctForJets,"forJets");
    iEvent.put(m_gctTauJets,"tauJets");
    iEvent.put(m_gctHfBitCounts);
    iEvent.put(m_gctHfRingEtSums);
    iEvent.put(m_gctEtTot);
    iEvent.put(m_gctEtHad);
    iEvent.put(m_gctEtMiss);
    iEvent.put(m_gctHtMiss);
    iEvent.put(m_gctJetCounts);  // Deprecated (empty collection still needed by GT)
  }

  if (doObjectMap_) {
    std::auto_ptr<L1GlobalTriggerObjectMapRecord> gtObjectMapRecord(
        new L1GlobalTriggerObjectMapRecord() );
    iEvent.put(gtObjectMapRecord);
  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(HLTDummyCollections);
