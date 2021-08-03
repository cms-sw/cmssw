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
#include "FWCore/Framework/interface/EDProducer.h"

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
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
// --- GCT
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

// -- ObjectMap
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"

//
// class decleration
//

class HLTDummyCollections : public edm::EDProducer {
public:
  explicit HLTDummyCollections(const edm::ParameterSet&);
  ~HLTDummyCollections() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------

  std::string action_;
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
  action_ = iConfig.getParameter<std::string>("action");
  unpackZDC_ = iConfig.getParameter<bool>("UnpackZDC");
  ESdigiCollection_ = iConfig.getParameter<std::string>("ESdigiCollection");

  //  doEcal_           = ( action_ == "doEcal");
  doHcal_ = (action_ == "doHcal");
  doEcalPreshower_ = (action_ == "doEcalPreshower");
  doMuonDTDigis_ = (action_ == "doMuonDT");
  doMuonCSCDigis_ = (action_ == "doMuonCSC");
  doSiPixelDigis_ = (action_ == "doSiPixel");
  doSiStrip_ = (action_ == "doSiStrip");
  doObjectMap_ = (action_ == "doObjectMap");
  doGCT_ = (action_ == "doGCT");

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
    produces<edm::DetSetVector<PixelDigi> >();
  }

  if (doSiStrip_) {
    produces<edmNew::DetSetVector<SiStripCluster> >();
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
void HLTDummyCollections::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  /*
  if (doEcal_) {
    std::unique_ptr< edm::LazyGetter<EcalRecHit> > Ecalcollection( new edm::LazyGetter<EcalRecHit> );
    iEvent.put(std::move(Ecalcollection));
    } */

  if (doHcal_) {
    std::unique_ptr<HBHEDigiCollection> hbhe_prod(new HBHEDigiCollection());
    std::unique_ptr<HFDigiCollection> hf_prod(new HFDigiCollection());
    std::unique_ptr<HODigiCollection> ho_prod(new HODigiCollection());
    std::unique_ptr<HcalTrigPrimDigiCollection> htp_prod(new HcalTrigPrimDigiCollection());
    std::unique_ptr<HOTrigPrimDigiCollection> hotp_prod(new HOTrigPrimDigiCollection());
    iEvent.put(std::move(hbhe_prod));
    iEvent.put(std::move(hf_prod));
    iEvent.put(std::move(ho_prod));
    iEvent.put(std::move(htp_prod));
    iEvent.put(std::move(hotp_prod));
    if (unpackZDC_) {
      std::unique_ptr<ZDCDigiCollection> zdcprod(new ZDCDigiCollection());
      iEvent.put(std::move(zdcprod));
    }
  }

  if (doEcalPreshower_) {
    std::unique_ptr<ESDigiCollection> productDigis(new ESDigiCollection);
    iEvent.put(std::move(productDigis), ESdigiCollection_);
  }

  if (doMuonDTDigis_) {
    std::unique_ptr<DTDigiCollection> detectorProduct(new DTDigiCollection);
    std::unique_ptr<DTLocalTriggerCollection> triggerProduct(new DTLocalTriggerCollection);
    iEvent.put(std::move(detectorProduct));
    iEvent.put(std::move(triggerProduct));
  }

  if (doMuonCSCDigis_) {
    std::unique_ptr<CSCWireDigiCollection> wireProduct(new CSCWireDigiCollection);
    std::unique_ptr<CSCStripDigiCollection> stripProduct(new CSCStripDigiCollection);
    std::unique_ptr<CSCALCTDigiCollection> alctProduct(new CSCALCTDigiCollection);
    std::unique_ptr<CSCCLCTDigiCollection> clctProduct(new CSCCLCTDigiCollection);
    std::unique_ptr<CSCComparatorDigiCollection> comparatorProduct(new CSCComparatorDigiCollection);
    std::unique_ptr<CSCRPCDigiCollection> rpcProduct(new CSCRPCDigiCollection);
    std::unique_ptr<CSCCorrelatedLCTDigiCollection> corrlctProduct(new CSCCorrelatedLCTDigiCollection);

    iEvent.put(std::move(wireProduct), "MuonCSCWireDigi");
    iEvent.put(std::move(stripProduct), "MuonCSCStripDigi");
    iEvent.put(std::move(alctProduct), "MuonCSCALCTDigi");
    iEvent.put(std::move(clctProduct), "MuonCSCCLCTDigi");
    iEvent.put(std::move(comparatorProduct), "MuonCSCComparatorDigi");
    iEvent.put(std::move(rpcProduct), "MuonCSCRPCDigi");
    iEvent.put(std::move(corrlctProduct), "MuonCSCCorrelatedLCTDigi");
  }

  if (doSiPixelDigis_) {
    std::unique_ptr<edm::DetSetVector<PixelDigi> > SiPicollection(new edm::DetSetVector<PixelDigi>);
    iEvent.put(std::move(SiPicollection));
  }

  if (doSiStrip_) {
    std::unique_ptr<edmNew::DetSetVector<SiStripCluster> > SiStripcollection(new edmNew::DetSetVector<SiStripCluster>);
    iEvent.put(std::move(SiStripcollection));
  }

  if (doGCT_) {
    std::unique_ptr<L1GctEmCandCollection> m_gctIsoEm(new L1GctEmCandCollection);
    std::unique_ptr<L1GctEmCandCollection> m_gctNonIsoEm(new L1GctEmCandCollection);
    std::unique_ptr<L1GctJetCandCollection> m_gctCenJets(new L1GctJetCandCollection);
    std::unique_ptr<L1GctJetCandCollection> m_gctForJets(new L1GctJetCandCollection);
    std::unique_ptr<L1GctJetCandCollection> m_gctTauJets(new L1GctJetCandCollection);
    std::unique_ptr<L1GctHFBitCountsCollection> m_gctHfBitCounts(new L1GctHFBitCountsCollection);
    std::unique_ptr<L1GctHFRingEtSumsCollection> m_gctHfRingEtSums(new L1GctHFRingEtSumsCollection);
    std::unique_ptr<L1GctEtTotalCollection> m_gctEtTot(new L1GctEtTotalCollection);
    std::unique_ptr<L1GctEtHadCollection> m_gctEtHad(new L1GctEtHadCollection);
    std::unique_ptr<L1GctEtMissCollection> m_gctEtMiss(new L1GctEtMissCollection);
    std::unique_ptr<L1GctHtMissCollection> m_gctHtMiss(new L1GctHtMissCollection);
    std::unique_ptr<L1GctJetCountsCollection> m_gctJetCounts(new L1GctJetCountsCollection);  // DEPRECATED

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
    std::unique_ptr<L1GlobalTriggerObjectMapRecord> gtObjectMapRecord(new L1GlobalTriggerObjectMapRecord());
    iEvent.put(std::move(gtObjectMapRecord));
  }
}

// declare this class as a framework plugin
DEFINE_FWK_MODULE(HLTDummyCollections);
