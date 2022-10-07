// -*- C++ -*-
//
// Package:    PhysicsTools/NanoAOD
// Class:      EGMSeedGainProducer
//
/**\class EGMSeedGainProducer EGMSeedGainProducer.cc PhysicsTools/NanoAOD/plugins/EGMSeedGainProducer.cc
 Description: [one line class summary]
 Implementation:
     [Notes on implementation]
*/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"

#include "DataFormats/Common/interface/View.h"

//
// class declaration
//

template <typename T>
class EGMSeedGainProducer : public edm::global::EDProducer<> {
public:
  explicit EGMSeedGainProducer(const edm::ParameterSet& iConfig)
      : src_(consumes<edm::View<T>>(iConfig.getParameter<edm::InputTag>("src"))),
        recHitsEB_(consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("recHitsEB"))),
        recHitsEE_(consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("recHitsEE"))) {
    produces<edm::ValueMap<int>>();
  }
  ~EGMSeedGainProducer() override{};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------

  edm::EDGetTokenT<edm::View<T>> src_;
  edm::EDGetTokenT<EcalRecHitCollection> recHitsEB_;
  edm::EDGetTokenT<EcalRecHitCollection> recHitsEE_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// member functions
//

// ------------ method called to produce the data  ------------
template <typename T>
void EGMSeedGainProducer<T>::produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  auto src = iEvent.getHandle(src_);
  const auto& recHitsEBProd = iEvent.get(recHitsEB_);
  const auto& recHitsEEProd = iEvent.get(recHitsEE_);

  unsigned nSrc = src->size();
  std::vector<int> gainSeed(nSrc, 12);

  // determine gain of seed crystal as in RecoEgamma/EgammaTools/src/PhotonEnergyCalibrator.cc
  for (unsigned i = 0; i < nSrc; i++) {
    auto obj = src->ptrAt(i);
    auto detid = obj->superCluster()->seed()->seed();
    const auto& coll = obj->isEB() ? recHitsEBProd : recHitsEEProd;
    auto seed = coll.find(detid);
    if (seed != coll.end()) {
      if (seed->checkFlag(EcalRecHit::kHasSwitchToGain6))
        gainSeed[i] = 6;
      if (seed->checkFlag(EcalRecHit::kHasSwitchToGain1))
        gainSeed[i] = 1;
    }
  }

  std::unique_ptr<edm::ValueMap<int>> gainSeedV(new edm::ValueMap<int>());
  edm::ValueMap<int>::Filler fillerCorr(*gainSeedV);
  fillerCorr.insert(src, gainSeed.begin(), gainSeed.end());
  fillerCorr.fill();
  iEvent.put(std::move(gainSeedV));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
template <typename T>
void EGMSeedGainProducer<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src")->setComment("input physics object collection");
  desc.add<edm::InputTag>("recHitsEB", edm::InputTag("reducedEgamma", "reducedEBRecHits"))
      ->setComment("EB rechit collection");
  desc.add<edm::InputTag>("recHitsEE", edm::InputTag("reducedEgamma", "reducedEERecHits"))
      ->setComment("EE rechit collection");
  descriptions.addDefault(desc);
}

typedef EGMSeedGainProducer<pat::Electron> ElectronSeedGainProducer;
typedef EGMSeedGainProducer<pat::Photon> PhotonSeedGainProducer;

//define this as a plug-in
DEFINE_FWK_MODULE(ElectronSeedGainProducer);
DEFINE_FWK_MODULE(PhotonSeedGainProducer);
