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
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

// -- Ecal
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitComparison.h"
// -- Hcal
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

// -- Ecal Preshower
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
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
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
// --- GCT
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

// -- ObjectMap
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"

//
// class decleration
//

class HLTDummyCollections : public edm::global::EDProducer<> {
public:
  explicit HLTDummyCollections(const edm::ParameterSet&);
  ~HLTDummyCollections() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------

  //bool doEcal_ ;
  bool doHcal_;
  bool unpackZDC_;
  bool doEcalPreshower_;
  std::string ESdigiCollection_;
  bool doMuonDTDigis_;
  bool doMuonCSCDigis_;
  bool doSiPixelDigis_;
  bool doSiStrip_;
  bool doGCT_;
  bool doObjectMap_;
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
HLTDummyCollections::HLTDummyCollections(const edm::ParameterSet& iConfig) {
  auto action = iConfig.getParameter<std::string>("action");
  unpackZDC_ = iConfig.getParameter<bool>("UnpackZDC");
  ESdigiCollection_ = iConfig.getParameter<std::string>("ESdigiCollection");

  //  doEcal_           = ( action == "doEcal");
  doHcal_ = (action == "doHcal");
  doEcalPreshower_ = (action == "doEcalPreshower");
  doMuonDTDigis_ = (action == "doMuonDT");
  doMuonCSCDigis_ = (action == "doMuonCSC");
  doSiPixelDigis_ = (action == "doSiPixel");
  doSiStrip_ = (action == "doSiStrip");
  doObjectMap_ = (action == "doObjectMap");
  doGCT_ = (action == "doGCT");

  /* This interface is out of data and I do not know what is the proper replacement
  if (doEcal_) {
    // ECAL unpacking :
    produces< edm::LazyGetter<EcalRecHit> >();
  } */

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
    produces<edm::DetSetVector<PixelDigi>>();
  }

  if (doSiStrip_) {
    produces<edmNew::DetSetVector<SiStripCluster>>();
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

HLTDummyCollections::~HLTDummyCollections() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

void HLTDummyCollections::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("action", "");
  desc.add<bool>("UnpackZDC", false);
  desc.add<std::string>("ESdigiCollection", "");
  descriptions.add("HLTDummyCollections", desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void HLTDummyCollections::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;

  /*
  if (doEcal_) {
    std::unique_ptr< edm::LazyGetter<EcalRecHit> > Ecalcollection( new edm::LazyGetter<EcalRecHit> );
    iEvent.put(std::move(Ecalcollection));
    } */

  if (doHcal_) {
    std::unique_ptr<HBHEDigiCollection> hbhe_prod = std::make_unique<HBHEDigiCollection>();
    std::unique_ptr<HFDigiCollection> hf_prod = std::make_unique<HFDigiCollection>();
    std::unique_ptr<HODigiCollection> ho_prod = std::make_unique<HODigiCollection>();
    std::unique_ptr<HcalTrigPrimDigiCollection> htp_prod = std::make_unique<HcalTrigPrimDigiCollection>();
    std::unique_ptr<HOTrigPrimDigiCollection> hotp_prod = std::make_unique<HOTrigPrimDigiCollection>();
    iEvent.put(std::move(hbhe_prod));
    iEvent.put(std::move(hf_prod));
    iEvent.put(std::move(ho_prod));
    iEvent.put(std::move(htp_prod));
    iEvent.put(std::move(hotp_prod));
    if (unpackZDC_) {
      std::unique_ptr<ZDCDigiCollection> zdcprod = std::make_unique<ZDCDigiCollection>();
      iEvent.put(std::move(zdcprod));
    }
  }

  if (doEcalPreshower_) {
    std::unique_ptr<ESDigiCollection> productDigis = std::make_unique<ESDigiCollection>();
    iEvent.put(std::move(productDigis), ESdigiCollection_);
  }

  if (doMuonDTDigis_) {
    std::unique_ptr<DTDigiCollection> detectorProduct = std::make_unique<DTDigiCollection>();
    std::unique_ptr<DTLocalTriggerCollection> triggerProduct = std::make_unique<DTLocalTriggerCollection>();
    iEvent.put(std::move(detectorProduct));
    iEvent.put(std::move(triggerProduct));
  }

  if (doMuonCSCDigis_) {
    std::unique_ptr<CSCWireDigiCollection> wireProduct = std::make_unique<CSCWireDigiCollection>();
    std::unique_ptr<CSCStripDigiCollection> stripProduct = std::make_unique<CSCStripDigiCollection>();
    std::unique_ptr<CSCALCTDigiCollection> alctProduct = std::make_unique<CSCALCTDigiCollection>();
    std::unique_ptr<CSCCLCTDigiCollection> clctProduct = std::make_unique<CSCCLCTDigiCollection>();
    std::unique_ptr<CSCComparatorDigiCollection> comparatorProduct = std::make_unique<CSCComparatorDigiCollection>();
    std::unique_ptr<CSCRPCDigiCollection> rpcProduct = std::make_unique<CSCRPCDigiCollection>();
    std::unique_ptr<CSCCorrelatedLCTDigiCollection> corrlctProduct = std::make_unique<CSCCorrelatedLCTDigiCollection>();

    iEvent.put(std::move(wireProduct), "MuonCSCWireDigi");
    iEvent.put(std::move(stripProduct), "MuonCSCStripDigi");
    iEvent.put(std::move(alctProduct), "MuonCSCALCTDigi");
    iEvent.put(std::move(clctProduct), "MuonCSCCLCTDigi");
    iEvent.put(std::move(comparatorProduct), "MuonCSCComparatorDigi");
    iEvent.put(std::move(rpcProduct), "MuonCSCRPCDigi");
    iEvent.put(std::move(corrlctProduct), "MuonCSCCorrelatedLCTDigi");
  }

  if (doSiPixelDigis_) {
    std::unique_ptr<edm::DetSetVector<PixelDigi>> SiPicollection = std::make_unique<edm::DetSetVector<PixelDigi>>();
    iEvent.put(std::move(SiPicollection));
  }

  if (doSiStrip_) {
    std::unique_ptr<edmNew::DetSetVector<SiStripCluster>> SiStripcollection =
        std::make_unique<edmNew::DetSetVector<SiStripCluster>>();
    iEvent.put(std::move(SiStripcollection));
  }

  if (doGCT_) {
    std::unique_ptr<L1GctEmCandCollection> m_gctIsoEm = std::make_unique<L1GctEmCandCollection>();
    std::unique_ptr<L1GctEmCandCollection> m_gctNonIsoEm = std::make_unique<L1GctEmCandCollection>();
    std::unique_ptr<L1GctJetCandCollection> m_gctCenJets = std::make_unique<L1GctJetCandCollection>();
    std::unique_ptr<L1GctJetCandCollection> m_gctForJets = std::make_unique<L1GctJetCandCollection>();
    std::unique_ptr<L1GctJetCandCollection> m_gctTauJets = std::make_unique<L1GctJetCandCollection>();
    std::unique_ptr<L1GctHFBitCountsCollection> m_gctHfBitCounts = std::make_unique<L1GctHFBitCountsCollection>();
    std::unique_ptr<L1GctHFRingEtSumsCollection> m_gctHfRingEtSums = std::make_unique<L1GctHFRingEtSumsCollection>();
    std::unique_ptr<L1GctEtTotalCollection> m_gctEtTot = std::make_unique<L1GctEtTotalCollection>();
    std::unique_ptr<L1GctEtHadCollection> m_gctEtHad = std::make_unique<L1GctEtHadCollection>();
    std::unique_ptr<L1GctEtMissCollection> m_gctEtMiss = std::make_unique<L1GctEtMissCollection>();
    std::unique_ptr<L1GctHtMissCollection> m_gctHtMiss = std::make_unique<L1GctHtMissCollection>();
    std::unique_ptr<L1GctJetCountsCollection> m_gctJetCounts =
        std::make_unique<L1GctJetCountsCollection>();  // DEPRECATED

    iEvent.put(std::move(m_gctIsoEm), "isoEm");
    iEvent.put(std::move(m_gctNonIsoEm), "nonIsoEm");
    iEvent.put(std::move(m_gctCenJets), "cenJets");
    iEvent.put(std::move(m_gctForJets), "forJets");
    iEvent.put(std::move(m_gctTauJets), "tauJets");
    iEvent.put(std::move(m_gctHfBitCounts));
    iEvent.put(std::move(m_gctHfRingEtSums));
    iEvent.put(std::move(m_gctEtTot));
    iEvent.put(std::move(m_gctEtHad));
    iEvent.put(std::move(m_gctEtMiss));
    iEvent.put(std::move(m_gctHtMiss));
    iEvent.put(std::move(m_gctJetCounts));  // Deprecated (empty collection still needed by GT)
  }

  if (doObjectMap_) {
    std::unique_ptr<L1GlobalTriggerObjectMapRecord> gtObjectMapRecord =
        std::make_unique<L1GlobalTriggerObjectMapRecord>();
    iEvent.put(std::move(gtObjectMapRecord));
  }
}

// declare this class as a framework plugin
DEFINE_FWK_MODULE(HLTDummyCollections);
