// system include files
#include <memory>
#include <string>
#include <iostream>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
//#include "Calibration/EcalAlCaRecoProducers/interface/AlCaElectronsProducer.h"
#include "DataFormats/EgammaCandidates/interface/SiStripElectron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"

#include "TFile.h"
#include "TH1.h"
#include "TH2.h"

#include <Math/VectorUtil.h>

class AlCaElectronsTest : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit AlCaElectronsTest(const edm::ParameterSet&);
  ~AlCaElectronsTest() = default;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  void beginJob() override;

private:
  EcalRecHit getMaximum(const EcalRecHitCollection* recHits);
  void fillAroundBarrel(const EcalRecHitCollection* recHits, int eta, int phi);
  void fillAroundEndcap(const EcalRecHitCollection* recHits, int ics, int ips);

  edm::EDGetTokenT<EBRecHitCollection> m_barrelAlCa;
  edm::EDGetTokenT<EERecHitCollection> m_endcapAlCa;
  std::string m_outputFileName;

  //! ECAL map
  TH2F* m_barrelGlobalCrystalsMap;
  //! local map
  TH2F* m_barrelLocalCrystalsMap;
  //! ECAL map
  TH2F* m_endcapGlobalCrystalsMap;
  //! local map
  TH2F* m_endcapLocalCrystalsMap;
  //! ECAL Energy
  TH2F* m_barrelGlobalCrystalsEnergy;
  //! local Energy
  TH2F* m_barrelLocalCrystalsEnergy;
  //! ECAL Energy
  TH2F* m_endcapGlobalCrystalsEnergy;
  //! local Energy
  TH2F* m_endcapLocalCrystalsEnergy;
  //! ECAL EnergyMap
  TH2F* m_barrelGlobalCrystalsEnergyMap;
  //! ECAL EnergyMap
  TH2F* m_endcapGlobalCrystalsEnergyMap;
};

// ----------------------------------------------------------------

AlCaElectronsTest::AlCaElectronsTest(const edm::ParameterSet& iConfig)
    : m_barrelAlCa(consumes<EBRecHitCollection>(iConfig.getParameter<edm::InputTag>("alcaBarrelHitCollection"))),
      m_endcapAlCa(consumes<EERecHitCollection>(iConfig.getParameter<edm::InputTag>("alcaEndcapHitCollection"))),
      m_outputFileName(
          iConfig.getUntrackedParameter<std::string>("HistOutFile", std::string("AlCaElectronsTest.root"))) {
  usesResource(TFileService::kSharedResource);
}

// ----------------------------------------------------------------

void AlCaElectronsTest::beginJob() {
  edm::Service<TFileService> fs;
  m_barrelGlobalCrystalsMap =
      fs->make<TH2F>("m_barrelGlobalCrystalsMap", "m_barrelGlobalCrystalsMap", 171, -85, 86, 360, 0, 360);
  m_barrelLocalCrystalsMap =
      fs->make<TH2F>("m_barrelLocalCrystalsMap", "m_barrelLocalCrystalsMap", 20, -10, 10, 20, -10, 10);
  m_endcapGlobalCrystalsMap =
      fs->make<TH2F>("m_endcapGlobalCrystalsMap", "m_endcapGlobalCrystalsMap", 100, 0, 100, 100, 0, 100);
  m_endcapLocalCrystalsMap =
      fs->make<TH2F>("m_endcapLocalCrystalsMap", "m_endcapLocalCrystalsMap", 20, -10, 10, 20, -10, 10);
  m_barrelGlobalCrystalsEnergy =
      fs->make<TH2F>("m_barrelGlobalCrystalsEnergy", "m_barrelGlobalCrystalsEnergy", 171, -85, 86, 360, 0, 360);
  m_barrelLocalCrystalsEnergy =
      fs->make<TH2F>("m_barrelLocalCrystalsEnergy", "m_barrelLocalCrystalsEnergy", 20, -10, 10, 20, -10, 10);
  m_endcapGlobalCrystalsEnergy =
      fs->make<TH2F>("m_endcapGlobalCrystalsEnergy", "m_endcapGlobalCrystalsEnergy", 100, 0, 100, 100, 0, 100);
  m_endcapLocalCrystalsEnergy =
      fs->make<TH2F>("m_endcapLocalCrystalsEnergy", "m_endcapLocalCrystalsEnergy", 20, -10, 10, 20, -10, 10);
  m_barrelGlobalCrystalsEnergyMap =
      fs->make<TH2F>("m_barrelGlobalCrystalsEnergyMap", "m_barrelGlobalCrystalsEnergyMap", 171, -85, 86, 360, 0, 360);
  m_endcapGlobalCrystalsEnergyMap =
      fs->make<TH2F>("m_endcapGlobalCrystalsEnergyMap", "m_endcapGlobalCrystalsEnergyMap", 100, 0, 100, 100, 0, 100);
}

// ----------------------------------------------------------------

void AlCaElectronsTest::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //FIXME replace with msg logger
  edm::LogVerbatim("ElectronsTest") << "[AlCaElectronsTest] analysing event " << iEvent.id();

  //PG get the collections
  // get Barrel RecHits
  edm::Handle<EBRecHitCollection> barrelRecHitsHandle;
  iEvent.getByToken(m_barrelAlCa, barrelRecHitsHandle);
  if (!barrelRecHitsHandle.isValid()) {
    edm::EDConsumerBase::Labels labels;
    labelsForToken(m_barrelAlCa, labels);
    edm::LogError("ElectronsTest") << "[AlCaElectronsTest] caught std::exception in rertieving " << labels.module;
    return;
  } else {
    const EBRecHitCollection* barrelHitsCollection = barrelRecHitsHandle.product();
    //PG fill the histo with the maximum
    EcalRecHit barrelMax = getMaximum(barrelHitsCollection);
    EBDetId barrelMaxId(barrelMax.id());
    m_barrelGlobalCrystalsMap->Fill(barrelMaxId.ieta(), barrelMaxId.iphi());
    m_barrelGlobalCrystalsEnergy->Fill(barrelMaxId.ieta(), barrelMaxId.iphi(), barrelMax.energy());
    fillAroundBarrel(barrelHitsCollection, barrelMaxId.ieta(), barrelMaxId.iphi());
  }

  // get Endcap RecHits
  edm::Handle<EERecHitCollection> endcapRecHitsHandle;
  iEvent.getByToken(m_endcapAlCa, endcapRecHitsHandle);
  if (!endcapRecHitsHandle.isValid()) {
    edm::EDConsumerBase::Labels labels;
    labelsForToken(m_endcapAlCa, labels);
    edm::LogError("ElectronsTest") << "[AlCaElectronsTest] caught std::exception in rertieving " << labels.module;
    return;
  } else {
    const EERecHitCollection* endcapHitsCollection = endcapRecHitsHandle.product();
    //PG fill the histo with the maximum
    EcalRecHit endcapMax = getMaximum(endcapHitsCollection);
    EEDetId endcapMaxId(endcapMax.id());
    m_endcapGlobalCrystalsMap->Fill(endcapMaxId.ix(), endcapMaxId.iy());
    m_endcapGlobalCrystalsEnergy->Fill(endcapMaxId.ix(), endcapMaxId.iy(), endcapMax.energy());
    fillAroundEndcap(endcapHitsCollection, endcapMaxId.ix(), endcapMaxId.iy());
  }
}

// ----------------------------------------------------------------

EcalRecHit AlCaElectronsTest::getMaximum(const EcalRecHitCollection* recHits) {
  double energy = 0.;
  EcalRecHit max;
  for (EcalRecHitCollection::const_iterator elem = recHits->begin(); elem != recHits->end(); ++elem) {
    if (elem->energy() > energy) {
      energy = elem->energy();
      max = *elem;
    }
  }
  return max;
}

// ----------------------------------------------------------------

void AlCaElectronsTest::fillAroundBarrel(const EcalRecHitCollection* recHits, int eta, int phi) {
  for (EcalRecHitCollection::const_iterator elem = recHits->begin(); elem != recHits->end(); ++elem) {
    EBDetId elementId = elem->id();
    m_barrelLocalCrystalsMap->Fill(elementId.ieta() - eta, elementId.iphi() - phi);
    m_barrelLocalCrystalsEnergy->Fill(elementId.ieta() - eta, elementId.iphi() - phi, elem->energy());
    m_barrelGlobalCrystalsEnergyMap->Fill(elementId.ieta(), elementId.iphi(), elem->energy());
  }
  return;
}

// ----------------------------------------------------------------

void AlCaElectronsTest::fillAroundEndcap(const EcalRecHitCollection* recHits, int ics, int ips) {
  for (EcalRecHitCollection::const_iterator elem = recHits->begin(); elem != recHits->end(); ++elem) {
    EEDetId elementId = elem->id();
    m_endcapLocalCrystalsMap->Fill(elementId.ix() - ics, elementId.iy() - ips);
    m_endcapLocalCrystalsEnergy->Fill(elementId.ix() - ics, elementId.iy() - ips, elem->energy());
    m_endcapGlobalCrystalsEnergyMap->Fill(elementId.ix(), elementId.iy(), elem->energy());
  }
  return;
}
