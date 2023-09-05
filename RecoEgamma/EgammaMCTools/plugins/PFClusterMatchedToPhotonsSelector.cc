// -*- C++ -*-
//
// Package:    CommonTools/RecoAlgos
// Class:      PFClusterMatchedToPhotonsSelector
//
/**\class PFClusterMatchedToPhotonsSelector PFClusterMatchedToPhotonsSelector.cc CommonTools/RecoAlgos/plugins/PFClusterMatchedToPhotonsSelector.cc

 Description: Matches ECAL PF clusters to photons that do not convert

 Implementation:

*/
//
// Original Author:  RCLSA
//         Created:  Wed, 22 Mar 2017 18:01:40 GMT
//
//

// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"

typedef reco::PFCluster::EEtoPSAssociation::value_type EEPSPair;
bool sortByKey(const EEPSPair& a, const EEPSPair& b) { return a.first < b.first; }

class PFClusterMatchedToPhotonsSelector : public edm::stream::EDProducer<> {
public:
  PFClusterMatchedToPhotonsSelector(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  edm::EDGetTokenT<reco::GenParticleCollection> genParticleToken_;  // genParticles
  edm::EDGetTokenT<reco::PFClusterCollection> particleFlowClusterECALToken_;
  edm::EDGetTokenT<reco::PFCluster::EEtoPSAssociation> associationToken_;
  edm::EDGetTokenT<TrackingParticleCollection> trackingParticleToken_;
  edm::EDGetTokenT<EcalRecHitCollection> recHitsEB_;
  edm::EDGetTokenT<EcalRecHitCollection> recHitsEE_;

  const EcalClusterLazyTools::ESGetTokens ecalClusterToolsESGetTokens_;

  double matchMaxDR2_;
  double matchMaxDEDR2_;

  double volumeZ_EB_;
  double volumeRadius_EB_;
  double volumeZ_EE_;
};

PFClusterMatchedToPhotonsSelector::PFClusterMatchedToPhotonsSelector(const edm::ParameterSet& iConfig)
    : ecalClusterToolsESGetTokens_{consumesCollector()} {
  //now do what ever initialization is needed
  particleFlowClusterECALToken_ =
      consumes<reco::PFClusterCollection>(iConfig.getParameter<edm::InputTag>("pfClustersTag"));
  associationToken_ =
      consumes<reco::PFCluster::EEtoPSAssociation>(iConfig.getParameter<edm::InputTag>("pfClustersTag"));
  trackingParticleToken_ =
      consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>("trackingParticleTag"));
  genParticleToken_ = consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("genParticleTag"));
  recHitsEB_ = consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("recHitsEBLabel"));
  recHitsEE_ = consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("recHitsEELabel"));

  matchMaxDR2_ = iConfig.getParameter<double>("maxDR2");
  matchMaxDEDR2_ = iConfig.getParameter<double>("maxDEDR2");
  volumeZ_EB_ = iConfig.getParameter<double>("volumeZ_EB");
  volumeRadius_EB_ = iConfig.getParameter<double>("volumeRadius_EB");
  volumeZ_EE_ = iConfig.getParameter<double>("volumeZ_EE");

  produces<reco::PFClusterCollection>();
  produces<edm::ValueMap<reco::GenParticleRef> >();
  produces<edm::ValueMap<int> >();
  produces<edm::ValueMap<float> >("PS1");
  produces<edm::ValueMap<float> >("PS2");
}

void PFClusterMatchedToPhotonsSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pfClustersTag", edm::InputTag("particleFlowClusterECAL"));
  desc.add<edm::InputTag>("trackingParticleTag", edm::InputTag("mix", "MergedTrackTruth"));
  desc.add<edm::InputTag>("genParticleTag", edm::InputTag("genParticles"));
  desc.add<edm::InputTag>("recHitsEBLabel", edm::InputTag("ecalRecHit", "EcalRecHitsEB"));
  desc.add<edm::InputTag>("recHitsEELabel", edm::InputTag("ecalRecHit", "EcalRecHitsEE"));
  desc.add<double>("maxDR2", 0.1 * 0.1);
  desc.add<double>("maxDEDR2", 0.5 * 0.5);
  desc.add<double>("volumeZ_EB", 304.5);
  desc.add<double>("volumeRadius_EB", 123.8);
  desc.add<double>("volumeZ_EE", 317.0);
  descriptions.add("pfClusterMatchedToPhotonsSelector", desc);
}

void PFClusterMatchedToPhotonsSelector::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<reco::PFClusterCollection> particleFlowClusterECALHandle_;
  edm::Handle<reco::PFCluster::EEtoPSAssociation> associationHandle_;
  edm::Handle<TrackingParticleCollection> trackingParticleHandle_;
  edm::Handle<reco::GenParticleCollection> genParticleHandle_;

  iEvent.getByToken(particleFlowClusterECALToken_, particleFlowClusterECALHandle_);
  iEvent.getByToken(trackingParticleToken_, trackingParticleHandle_);
  iEvent.getByToken(genParticleToken_, genParticleHandle_);
  iEvent.getByToken(associationToken_, associationHandle_);

  std::unique_ptr<reco::PFClusterCollection> out = std::make_unique<reco::PFClusterCollection>();
  std::unique_ptr<edm::ValueMap<reco::GenParticleRef> > genmatching_out =
      std::make_unique<edm::ValueMap<reco::GenParticleRef> >();
  std::unique_ptr<edm::ValueMap<int> > clustersize_out = std::make_unique<edm::ValueMap<int> >();
  std::unique_ptr<edm::ValueMap<float> > energyPS1_out = std::make_unique<edm::ValueMap<float> >();
  std::unique_ptr<edm::ValueMap<float> > energyPS2_out = std::make_unique<edm::ValueMap<float> >();

  std::vector<reco::GenParticleRef> genmatching;
  std::vector<int> clustersize;
  std::vector<float> energyPS1;
  std::vector<float> energyPS2;

  EcalClusterLazyTools lazyTool(iEvent, ecalClusterToolsESGetTokens_.get(iSetup), recHitsEB_, recHitsEE_);

  for (size_t iP = 0; iP < particleFlowClusterECALHandle_->size(); iP++) {
    auto&& pfCluster = particleFlowClusterECALHandle_->at(iP);
    bool isMatched = false;
    reco::GenParticleRef::key_type matchedKey;

    // Preselect PFclusters
    if (pfCluster.energy() <= 0) {
      continue;
    }

    for (auto&& trackingParticle : *trackingParticleHandle_) {
      if (trackingParticle.pdgId() != 22)
        continue;
      if (trackingParticle.status() != 1)
        continue;
      matchedKey = trackingParticle.genParticles().at(0).key();

      float dR2 = reco::deltaR2(trackingParticle, pfCluster.position());
      if (dR2 > matchMaxDR2_)
        continue;
      float dE = 1. - trackingParticle.genParticles().at(0)->energy() / pfCluster.energy();
      if ((dR2 + dE * dE) > matchMaxDEDR2_)
        continue;

      bool isConversion = false;
      for (auto&& vertRef : trackingParticle.decayVertices()) {
        if (vertRef->position().rho() > volumeRadius_EB_ && std::abs(vertRef->position().z()) < volumeZ_EB_)
          continue;
        if (std::abs(vertRef->position().z()) > volumeZ_EE_)
          continue;

        for (auto&& tpRef : vertRef->daughterTracks()) {
          if (std::abs(tpRef->pdgId()) == 11)
            isConversion = true;
          break;
        }
        if (isConversion)
          break;
      }
      if (isConversion)
        continue;

      isMatched = true;
      break;
    }

    if (isMatched) {
      out->push_back(pfCluster);
      double ePS1 = 0, ePS2 = 0;
      if (!(pfCluster.layer() == PFLayer::ECAL_BARREL)) {
        auto ee_key_val = std::make_pair(iP, edm::Ptr<reco::PFCluster>());
        const auto clustops = std::equal_range(
            associationHandle_.product()->begin(), associationHandle_.product()->end(), ee_key_val, sortByKey);
        for (auto i_ps = clustops.first; i_ps != clustops.second; ++i_ps) {
          edm::Ptr<reco::PFCluster> psclus(i_ps->second);
          switch (psclus->layer()) {
            case PFLayer::PS1:
              ePS1 += psclus->energy();
              break;
            case PFLayer::PS2:
              ePS2 += psclus->energy();
              break;
            default:
              break;
          }
        }
      }

      genmatching.push_back(edm::Ref<reco::GenParticleCollection>(genParticleHandle_, matchedKey));
      clustersize.push_back(lazyTool.n5x5(pfCluster));
      energyPS1.push_back(ePS1);
      energyPS2.push_back(ePS2);
    }
  }

  edm::OrphanHandle<reco::PFClusterCollection> pfClusterHandle = iEvent.put(std::move(out));

  edm::ValueMap<reco::GenParticleRef>::Filler mapFiller(*genmatching_out);
  mapFiller.insert(pfClusterHandle, genmatching.begin(), genmatching.end());
  mapFiller.fill();
  iEvent.put(std::move(genmatching_out));

  edm::ValueMap<int>::Filler mapFiller_int(*clustersize_out);
  mapFiller_int.insert(pfClusterHandle, clustersize.begin(), clustersize.end());
  mapFiller_int.fill();
  iEvent.put(std::move(clustersize_out));

  edm::ValueMap<float>::Filler mapFiller_energyPS1(*energyPS1_out);
  mapFiller_energyPS1.insert(pfClusterHandle, energyPS1.begin(), energyPS1.end());
  mapFiller_energyPS1.fill();
  iEvent.put(std::move(energyPS1_out), "PS1");

  edm::ValueMap<float>::Filler mapFiller_energyPS2(*energyPS2_out);
  mapFiller_energyPS2.insert(pfClusterHandle, energyPS2.begin(), energyPS2.end());
  mapFiller_energyPS2.fill();
  iEvent.put(std::move(energyPS2_out), "PS2");
}

//define this as a plug-in
DEFINE_FWK_MODULE(PFClusterMatchedToPhotonsSelector);
