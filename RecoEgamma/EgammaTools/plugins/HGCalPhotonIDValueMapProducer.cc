// -*- C++ -*-
//
// Package:   RecoEgamma/EgammaTools
// Class:    HGCalPhotonIDValueMapProducer
//
/**\class HGCalPhotonIDValueMapProducer HGCalPhotonIDValueMapProducer.cc RecoEgamma/EgammaTools/plugins/HGCalPhotonIDValueMapProducer.cc

 Description: [one line class summary]

 Implementation:
    [Notes on implementation]
*/
//
// Original Author:  Nicholas Charles Smith
//      Created:  Wed, 05 Apr 2017 12:17:43 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

#include "RecoEgamma/EgammaTools/interface/HGCalEgammaIDHelper.h"
#include "RecoEgamma/EgammaTools/interface/LongDeps.h"

class HGCalPhotonIDValueMapProducer : public edm::stream::EDProducer<> {
public:
  explicit HGCalPhotonIDValueMapProducer(const edm::ParameterSet&);
  ~HGCalPhotonIDValueMapProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<edm::View<reco::Photon>> photonsToken_;
  float radius_;

  static const std::vector<std::string> valuesProduced_;
  std::map<const std::string, std::vector<float>> maps_;

  std::unique_ptr<HGCalEgammaIDHelper> phoIDHelper_;
};

// All the ValueMap names to output are defined in the auto-generated python cfi
// so that potential consumers can configure themselves in a simple manner
// Would be cool to use compile-time validation, but need constexpr strings, e.g. std::string_view in C++17
const std::vector<std::string> HGCalPhotonIDValueMapProducer::valuesProduced_ = {
    "seedEt",       "seedEnergy",    "seedEnergyEE",  "seedEnergyFH",  "seedEnergyBH",
    "pcaEig1",      "pcaEig2",       "pcaEig3",       "pcaSig1",       "pcaSig2",
    "pcaSig3",      "sigmaUU",       "sigmaVV",       "sigmaEE",       "sigmaPP",
    "nLayers",      "firstLayer",    "lastLayer",     "e4oEtot",       "layerEfrac10",
    "layerEfrac90", "measuredDepth", "expectedDepth", "expectedSigma", "depthCompatibility",
    "caloIsoRing0", "caloIsoRing1",  "caloIsoRing2",  "caloIsoRing3",  "caloIsoRing4",
};

HGCalPhotonIDValueMapProducer::HGCalPhotonIDValueMapProducer(const edm::ParameterSet& iConfig)
    : photonsToken_(consumes(iConfig.getParameter<edm::InputTag>("photons"))),
      radius_(iConfig.getParameter<double>("pcaRadius")) {
  for (const auto& key : valuesProduced_) {
    maps_[key] = {};
    produces<edm::ValueMap<float>>(key);
  }

  phoIDHelper_ = std::make_unique<HGCalEgammaIDHelper>(iConfig, consumesCollector());
}

HGCalPhotonIDValueMapProducer::~HGCalPhotonIDValueMapProducer() {}

// ------------ method called to produce the data  ------------
void HGCalPhotonIDValueMapProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  auto photonsH = iEvent.getHandle(photonsToken_);

  // Clear previous map
  for (auto&& kv : maps_) {
    kv.second.clear();
    kv.second.reserve(photonsH->size());
  }

  // Set up helper tool
  phoIDHelper_->eventInit(iEvent, iSetup);

  for (const auto& pho : *photonsH) {
    if (pho.isEB()) {
      // Fill some dummy value
      for (auto&& kv : maps_) {
        kv.second.push_back(0.);
      }
    } else {
      phoIDHelper_->computeHGCAL(pho, radius_);

      // check the PCA has worked out
      if (phoIDHelper_->sigmaUU() == -1) {
        for (auto&& kv : maps_) {
          kv.second.push_back(0.);
        }
        continue;
      }

      hgcal::LongDeps ld(phoIDHelper_->energyPerLayer(radius_, true));
      float measuredDepth, expectedDepth, expectedSigma;
      float depthCompatibility =
          phoIDHelper_->clusterDepthCompatibility(ld, measuredDepth, expectedDepth, expectedSigma);

      // Fill here all the ValueMaps from their appropriate functions

      // energies calculated in an cylinder around the axis of the pho cluster
      float seed_tot_energy = ld.energyEE() + ld.energyFH() + ld.energyBH();
      const double div_cosh_eta =
          pho.superCluster()->seed()->position().rho() / pho.superCluster()->seed()->position().r();
      maps_["seedEt"].push_back(seed_tot_energy * div_cosh_eta);
      maps_["seedEnergy"].push_back(seed_tot_energy);
      maps_["seedEnergyEE"].push_back(ld.energyEE());
      maps_["seedEnergyFH"].push_back(ld.energyFH());
      maps_["seedEnergyBH"].push_back(ld.energyBH());

      // Cluster shapes
      // PCA related
      maps_["pcaEig1"].push_back(phoIDHelper_->eigenValues()(0));
      maps_["pcaEig2"].push_back(phoIDHelper_->eigenValues()(1));
      maps_["pcaEig3"].push_back(phoIDHelper_->eigenValues()(2));
      maps_["pcaSig1"].push_back(phoIDHelper_->sigmas()(0));
      maps_["pcaSig2"].push_back(phoIDHelper_->sigmas()(1));
      maps_["pcaSig3"].push_back(phoIDHelper_->sigmas()(2));

      // transverse shapes
      maps_["sigmaUU"].push_back(phoIDHelper_->sigmaUU());
      maps_["sigmaVV"].push_back(phoIDHelper_->sigmaVV());
      maps_["sigmaEE"].push_back(phoIDHelper_->sigmaEE());
      maps_["sigmaPP"].push_back(phoIDHelper_->sigmaPP());

      // long profile
      maps_["nLayers"].push_back(ld.nLayers());
      maps_["firstLayer"].push_back(ld.firstLayer());
      maps_["lastLayer"].push_back(ld.lastLayer());
      maps_["e4oEtot"].push_back(ld.e4oEtot());
      maps_["layerEfrac10"].push_back(ld.layerEfrac10());
      maps_["layerEfrac90"].push_back(ld.layerEfrac90());

      // depth
      maps_["measuredDepth"].push_back(measuredDepth);
      maps_["expectedDepth"].push_back(expectedDepth);
      maps_["expectedSigma"].push_back(expectedSigma);
      maps_["depthCompatibility"].push_back(depthCompatibility);

      // Isolation
      maps_["caloIsoRing0"].push_back(phoIDHelper_->getIsolationRing(0));
      maps_["caloIsoRing1"].push_back(phoIDHelper_->getIsolationRing(1));
      maps_["caloIsoRing2"].push_back(phoIDHelper_->getIsolationRing(2));
      maps_["caloIsoRing3"].push_back(phoIDHelper_->getIsolationRing(3));
      maps_["caloIsoRing4"].push_back(phoIDHelper_->getIsolationRing(4));
    }
  }

  // Check we didn't make up a new variable and forget it in the constructor
  // (or some other pathology)
  if (maps_.size() != valuesProduced_.size()) {
    throw cms::Exception("HGCalPhotonIDValueMapProducer")
        << "We have a miscoded value map producer, since map size changed";
  }

  for (auto&& kv : maps_) {
    // Check we didn't forget any values
    if (kv.second.size() != photonsH->size()) {
      throw cms::Exception("HGCalPhotonIDValueMapProducer")
          << "We have a miscoded value map producer, since the variable " << kv.first << " wasn't filled.";
    }
    // Do the filling
    auto out = std::make_unique<edm::ValueMap<float>>();
    edm::ValueMap<float>::Filler filler(*out);
    filler.insert(photonsH, kv.second.begin(), kv.second.end());
    filler.fill();
    // and put it into the event
    iEvent.put(std::move(out), kv.first);
  }
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void HGCalPhotonIDValueMapProducer::beginStream(edm::StreamID) {}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void HGCalPhotonIDValueMapProducer::endStream() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HGCalPhotonIDValueMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // hgcalPhotonIDValueMap
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("photons", edm::InputTag("photonsFromMultiCl"));
  desc.add<double>("pcaRadius", 3.0);
  desc.add<std::vector<std::string>>("variables", valuesProduced_);
  desc.add<std::vector<double>>("dEdXWeights")
      ->setComment("This must be copied from dEdX_weights in RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi");
  desc.add<unsigned int>("isoNRings", 5);
  desc.add<double>("isoDeltaR", 0.15);
  desc.add<double>("isoDeltaRmin", 0.0);
  desc.add<edm::InputTag>("EERecHits", edm::InputTag("HGCalRecHit", "HGCEERecHits"));
  desc.add<edm::InputTag>("FHRecHits", edm::InputTag("HGCalRecHit", "HGCHEFRecHits"));
  desc.add<edm::InputTag>("BHRecHits", edm::InputTag("HGCalRecHit", "HGCHEBRecHits"));
  desc.add<edm::InputTag>("hitMapTag", edm::InputTag("hgcalRecHitMapProducer"));
  descriptions.add("hgcalPhotonIDValueMap", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalPhotonIDValueMapProducer);
