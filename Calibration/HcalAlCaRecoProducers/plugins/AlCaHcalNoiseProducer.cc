/*
Original author Grigory Safronov

27/03/09 - compilation from :
HLTrigger/special/src/HLTHcalNoiseFilter.cc
Calibration/HcalAlCaRecoProducers/src/AlCaEcalHcalReadoutsProducer.cc
Calibration/HcalIsolatedTrackReco/src/SubdetFEDSelector.cc

*/

// -*- C++ -*-

// system include files
#include <memory>
#include <string>
// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "EventFilter/RawDataCollector/interface/RawDataFEDSelector.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

//
// class decleration
//

class AlCaHcalNoiseProducer : public edm::one::EDProducer<> {
public:
  explicit AlCaHcalNoiseProducer(const edm::ParameterSet&);
  ~AlCaHcalNoiseProducer() override = default;

  void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // ----------member data ---------------------------

  bool useMet_;
  bool useJet_;
  double MetCut_;
  double JetMinE_;
  double JetHCALminEnergyFraction_;
  int nAnomalousEvents;
  int nEvents;

  std::vector<edm::InputTag> ecalLabels_;

  edm::EDGetTokenT<reco::CaloJetCollection> tok_jets_;
  edm::EDGetTokenT<reco::CaloMETCollection> tok_met_;
  edm::EDGetTokenT<CaloTowerCollection> tok_tower_;

  edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
  edm::EDGetTokenT<HORecHitCollection> tok_ho_;
  edm::EDGetTokenT<HFRecHitCollection> tok_hf_;

  edm::EDGetTokenT<EcalRecHitCollection> tok_ps_;
  edm::EDGetTokenT<FEDRawDataCollection> tok_raw_;
  std::vector<edm::EDGetTokenT<EcalRecHitCollection>> toks_ecal_;
};

AlCaHcalNoiseProducer::AlCaHcalNoiseProducer(const edm::ParameterSet& iConfig) {
  tok_jets_ = consumes<reco::CaloJetCollection>(iConfig.getParameter<edm::InputTag>("JetSource"));
  tok_met_ = consumes<reco::CaloMETCollection>(iConfig.getParameter<edm::InputTag>("MetSource"));
  tok_tower_ = consumes<CaloTowerCollection>(iConfig.getParameter<edm::InputTag>("TowerSource"));
  useMet_ = iConfig.getParameter<bool>("UseMET");
  useJet_ = iConfig.getParameter<bool>("UseJet");
  MetCut_ = iConfig.getParameter<double>("MetCut");
  JetMinE_ = iConfig.getParameter<double>("JetMinE");
  JetHCALminEnergyFraction_ = iConfig.getParameter<double>("JetHCALminEnergyFraction");

  tok_ho_ = consumes<HORecHitCollection>(iConfig.getParameter<edm::InputTag>("hoInput"));
  tok_hf_ = consumes<HFRecHitCollection>(iConfig.getParameter<edm::InputTag>("hfInput"));
  tok_hbhe_ = consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("hbheInput"));
  ecalLabels_ = iConfig.getParameter<std::vector<edm::InputTag>>("ecalInputs");
  tok_ps_ = consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("ecalPSInput"));
  tok_raw_ = consumes<FEDRawDataCollection>(iConfig.getParameter<edm::InputTag>("rawInput"));

  const unsigned nLabels = ecalLabels_.size();
  for (unsigned i = 0; i != nLabels; i++)
    toks_ecal_.push_back(consumes<EcalRecHitCollection>(ecalLabels_[i]));

  //register products
  produces<HBHERecHitCollection>("HBHERecHitCollectionFHN");
  produces<HORecHitCollection>("HORecHitCollectionFHN");
  produces<HFRecHitCollection>("HFRecHitCollectionFHN");

  produces<EcalRecHitCollection>("EcalRecHitCollectionFHN");
  produces<EcalRecHitCollection>("PSEcalRecHitCollectionFHN");

  produces<FEDRawDataCollection>("HcalFEDsFHN");
}

// ------------ method called to produce the data  ------------
void AlCaHcalNoiseProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  bool acceptEvent = false;

  // filtering basing on HLTrigger/special/src/HLTHcalNoiseFilter.cc:

  bool isAnomalous_BasedOnMET = false;
  bool isAnomalous_BasedOnEnergyFraction = false;

  if (useMet_) {
    edm::Handle<reco::CaloMETCollection> metHandle = iEvent.getHandle(tok_met_);
    const reco::CaloMETCollection* metCol = metHandle.product();
    const reco::CaloMET met = metCol->front();

    if (met.pt() > MetCut_)
      isAnomalous_BasedOnMET = true;
  }

  if (useJet_) {
    edm::Handle<reco::CaloJetCollection> calojetHandle = iEvent.getHandle(tok_jets_);
    edm::Handle<CaloTowerCollection> towerHandle = iEvent.getHandle(tok_tower_);

    std::vector<CaloTower> TowerContainer;
    std::vector<reco::CaloJet> JetContainer;
    TowerContainer.clear();
    JetContainer.clear();
    CaloTower seedTower;
    nEvents++;
    for (reco::CaloJetCollection::const_iterator calojetIter = calojetHandle->begin();
         calojetIter != calojetHandle->end();
         ++calojetIter) {
      if (((calojetIter->et()) * cosh(calojetIter->eta()) > JetMinE_) &&
          (calojetIter->energyFractionHadronic() > JetHCALminEnergyFraction_)) {
        JetContainer.push_back(*calojetIter);
        double maxTowerE = 0.0;
        for (CaloTowerCollection::const_iterator kal = towerHandle->begin(); kal != towerHandle->end(); kal++) {
          double dR = deltaR((*calojetIter).eta(), (*calojetIter).phi(), (*kal).eta(), (*kal).phi());
          if ((dR < 0.50) && (kal->p() > maxTowerE)) {
            maxTowerE = kal->p();
            seedTower = *kal;
          }
        }
        TowerContainer.push_back(seedTower);
      }
    }
    if (!JetContainer.empty()) {
      nAnomalousEvents++;
      isAnomalous_BasedOnEnergyFraction = true;
    }
  }

  acceptEvent = ((useMet_ && isAnomalous_BasedOnMET) || (useJet_ && isAnomalous_BasedOnEnergyFraction));

  ////////////////

  //Create empty output collections

  auto miniHBHERecHitCollection = std::make_unique<HBHERecHitCollection>();
  auto miniHORecHitCollection = std::make_unique<HORecHitCollection>();
  auto miniHFRecHitCollection = std::make_unique<HFRecHitCollection>();

  auto outputEColl = std::make_unique<EcalRecHitCollection>();
  auto outputESColl = std::make_unique<EcalRecHitCollection>();

  auto outputFEDs = std::make_unique<FEDRawDataCollection>();

  // if good event get and save all colletions
  if (acceptEvent) {
    edm::Handle<HBHERecHitCollection> hbhe = iEvent.getHandle(tok_hbhe_);
    edm::Handle<HORecHitCollection> ho = iEvent.getHandle(tok_ho_);
    edm::Handle<HFRecHitCollection> hf = iEvent.getHandle(tok_hf_);

    edm::Handle<EcalRecHitCollection> pRecHits = iEvent.getHandle(tok_ps_);

    // temporary collection of EB+EE recHits

    auto tmpEcalRecHitCollection = std::make_unique<EcalRecHitCollection>();

    std::vector<edm::EDGetTokenT<EcalRecHitCollection>>::const_iterator i;
    for (i = toks_ecal_.begin(); i != toks_ecal_.end(); i++) {
      edm::Handle<EcalRecHitCollection> ec = iEvent.getHandle(*i);
      for (EcalRecHitCollection::const_iterator recHit = (*ec).begin(); recHit != (*ec).end(); ++recHit) {
        tmpEcalRecHitCollection->push_back(*recHit);
      }
    }

    //////////

    //////// write HCAL collections:
    const HBHERecHitCollection Hithbhe = *(hbhe.product());
    for (HBHERecHitCollection::const_iterator hbheItr = Hithbhe.begin(); hbheItr != Hithbhe.end(); hbheItr++) {
      miniHBHERecHitCollection->push_back(*hbheItr);
    }
    const HORecHitCollection Hitho = *(ho.product());
    for (HORecHitCollection::const_iterator hoItr = Hitho.begin(); hoItr != Hitho.end(); hoItr++) {
      miniHORecHitCollection->push_back(*hoItr);
    }

    const HFRecHitCollection Hithf = *(hf.product());
    for (HFRecHitCollection::const_iterator hfItr = Hithf.begin(); hfItr != Hithf.end(); hfItr++) {
      miniHFRecHitCollection->push_back(*hfItr);
    }
    /////

    ///// write ECAL
    for (std::vector<EcalRecHit>::const_iterator ehit = tmpEcalRecHitCollection->begin();
         ehit != tmpEcalRecHitCollection->end();
         ehit++) {
      outputEColl->push_back(*ehit);
    }
    /////////

    // write PS
    const EcalRecHitCollection& psrechits = *(pRecHits.product());

    for (EcalRecHitCollection::const_iterator i = psrechits.begin(); i != psrechits.end(); i++) {
      outputESColl->push_back(*i);
    }

    // get HCAL FEDs
    edm::Handle<FEDRawDataCollection> rawIn = iEvent.getHandle(tok_raw_);

    std::vector<int> selFEDs;
    for (int i = FEDNumbering::MINHCALFEDID; i <= FEDNumbering::MAXHCALFEDID; i++) {
      selFEDs.push_back(i);
    }
    ////////////

    // Copying FEDs :
    const FEDRawDataCollection* rdc = rawIn.product();

    //   if ( ( rawData[i].provenance()->processName() != e.processHistory().rbegin()->processName() ) )
    //       continue ; // skip all raw collections not produced by the current process

    for (int j = 0; j <= FEDNumbering::MAXFEDID; ++j) {
      bool rightFED = false;
      for (uint32_t k = 0; k < selFEDs.size(); k++) {
        if (j == selFEDs[k]) {
          rightFED = true;
        }
      }
      if (!rightFED)
        continue;
      const FEDRawData& fedData = rdc->FEDData(j);
      size_t size = fedData.size();

      if (size > 0) {
        // this fed has data -- lets copy it
        FEDRawData& fedDataProd = outputFEDs->FEDData(j);
        if (fedDataProd.size() != 0) {
          edm::LogWarning("HcalNoise") << " More than one FEDRawDataCollection with data in FED " << j
                                       << " Skipping the 2nd\n";
          continue;
        }
        fedDataProd.resize(size);
        unsigned char* dataProd = fedDataProd.data();
        const unsigned char* data = fedData.data();
        for (unsigned int k = 0; k < size; ++k) {
          dataProd[k] = data[k];
        }
      }
    }
    //////////////////////
  }

  //Put selected information in the event
  iEvent.put(std::move(miniHBHERecHitCollection), "HBHERecHitCollectionFHN");
  iEvent.put(std::move(miniHORecHitCollection), "HORecHitCollectionFHN");
  iEvent.put(std::move(miniHFRecHitCollection), "HFRecHitCollectionFHN");
  iEvent.put(std::move(outputEColl), "EcalRecHitCollectionFHN");
  iEvent.put(std::move(outputESColl), "PSEcalRecHitCollectionFHN");
  iEvent.put(std::move(outputFEDs), "HcalFEDsFHN");
}

void AlCaHcalNoiseProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("JetSource", edm::InputTag("iterativeCone5CaloJets"));
  desc.add<edm::InputTag>("MetSource", edm::InputTag("met"));
  desc.add<edm::InputTag>("TowerSource", edm::InputTag("towerMaker"));
  desc.add<bool>("UseJet", true);
  desc.add<bool>("UseMET", false);
  desc.add<double>("MetCut", 0);
  desc.add<double>("JetMinE", 20);
  desc.add<double>("JetHCALminEnergyFraction", 0.98);
  desc.add<edm::InputTag>("hbheInput", edm::InputTag("hbhereco"));
  desc.add<edm::InputTag>("hfInput", edm::InputTag("hfreco"));
  desc.add<edm::InputTag>("hoInput", edm::InputTag("horeco"));
  std::vector<edm::InputTag> inputs = {edm::InputTag("ecalRecHit", "EcalRecHitsEB"),
                                       edm::InputTag("ecalRecHit", "EcalRecHitsEE")};
  desc.add<std::vector<edm::InputTag>>("ecalInputs", inputs);
  desc.add<edm::InputTag>("ecalPSInput", edm::InputTag("ecalPreshowerRecHit", "EcalRecHitsES"));
  desc.add<edm::InputTag>("rawInput", edm::InputTag("rawDataCollector"));
  descriptions.add("alcaHcalNoiseProducer", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(AlCaHcalNoiseProducer);
