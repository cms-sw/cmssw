#ifndef __CorrectedECALPFClusterProducer__
#define __CorrectedECALPFClusterProducer__

// user include files
#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterEMEnergyCorrector.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"
#include "DataFormats/Math/interface/deltaPhi.h"

namespace {
  typedef reco::PFCluster::EEtoPSAssociation::value_type EEPSPair;
  bool sortByKey(const EEPSPair& a, const EEPSPair& b) { return a.first < b.first; }

  // why, why -1 and not <double>::max() ????
  double testPreshowerDistance(reco::PFCluster const& eeclus, reco::PFCluster const& psclus) {
    auto const& pspos = psclus.positionREP();
    auto const& eepos = eeclus.positionREP();
    // lazy continue based on geometry
    if (eeclus.z() * psclus.z() < 0)
      return -1.0;
    auto deta = std::abs(eepos.eta() - pspos.eta());
    if (deta > 0.3)
      return -1.0;
    auto dphi = std::abs(deltaPhi(eepos.phi(), pspos.phi()));
    if (dphi > 0.6)
      return -1.0;
    return LinkByRecHit::testECALAndPSByRecHit(eeclus, psclus, false);
  }
}  // namespace

class CorrectedECALPFClusterProducer : public edm::stream::EDProducer<> {
public:
  CorrectedECALPFClusterProducer(const edm::ParameterSet& conf)
      : minimumPSEnergy_(conf.getParameter<double>("minimumPSEnergy")), skipPS_(conf.getParameter<bool>("skipPS")) {
    const edm::InputTag& inputECAL = conf.getParameter<edm::InputTag>("inputECAL");
    inputECAL_ = consumes<reco::PFClusterCollection>(inputECAL);

    const edm::InputTag& inputPS = conf.getParameter<edm::InputTag>("inputPS");
    if (!skipPS_)
      inputPS_ = consumes<reco::PFClusterCollection>(inputPS);

    const edm::ParameterSet& corConf = conf.getParameterSet("energyCorrector");
    corrector_ = std::make_unique<PFClusterEMEnergyCorrector>(corConf, consumesCollector());

    produces<reco::PFCluster::EEtoPSAssociation>();
    produces<reco::PFClusterCollection>();
  }

  void produce(edm::Event& e, const edm::EventSetup& es) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const double minimumPSEnergy_;
  const bool skipPS_;
  std::unique_ptr<PFClusterEMEnergyCorrector> corrector_;
  edm::EDGetTokenT<reco::PFClusterCollection> inputECAL_;
  edm::EDGetTokenT<reco::PFClusterCollection> inputPS_;
};

DEFINE_FWK_MODULE(CorrectedECALPFClusterProducer);

void CorrectedECALPFClusterProducer::produce(edm::Event& e, const edm::EventSetup& es) {
  auto clusters_out = std::make_unique<reco::PFClusterCollection>();
  auto association_out = std::make_unique<reco::PFCluster::EEtoPSAssociation>();

  edm::Handle<reco::PFClusterCollection> handleECAL;
  e.getByToken(inputECAL_, handleECAL);
  edm::Handle<reco::PFClusterCollection> handlePS;
  if (!skipPS_)
    e.getByToken(inputPS_, handlePS);

  auto const& ecals = *handleECAL;

  clusters_out->reserve(ecals.size());
  association_out->reserve(ecals.size());
  clusters_out->insert(clusters_out->end(), ecals.begin(), ecals.end());

  //build the EE->PS association
  if (!skipPS_) {
    auto const& pss = *handlePS;
    for (unsigned i = 0; i < pss.size(); ++i) {
      switch (pss[i].layer()) {  // just in case this isn't the ES...
        case PFLayer::PS1:
        case PFLayer::PS2:
          break;
        default:
          continue;
      }
      if (pss[i].energy() < minimumPSEnergy_)
        continue;
      int eematch = -1;
      auto min_dist = std::numeric_limits<double>::max();
      for (size_t ic = 0; ic < ecals.size(); ++ic) {
        if (ecals[ic].layer() != PFLayer::ECAL_ENDCAP)
          continue;
        auto dist = testPreshowerDistance(ecals[ic], pss[i]);
        if (dist == -1.0)
          dist = std::numeric_limits<double>::max();
        if (dist < min_dist) {
          eematch = ic;
          min_dist = dist;
        }
      }  // loop on EE clusters
      if (eematch >= 0) {
        edm::Ptr<reco::PFCluster> psclus(handlePS, i);
        association_out->push_back(std::make_pair(eematch, psclus));
      }
    }
  }
  std::sort(association_out->begin(), association_out->end(), sortByKey);

  corrector_->correctEnergies(e, es, *association_out, *clusters_out);

  association_out->shrink_to_fit();

  e.put(std::move(association_out));
  e.put(std::move(clusters_out));
}

void CorrectedECALPFClusterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("minimumPSEnergy", 0.0);
  desc.ifValue(
      edm::ParameterDescription<bool>("skipPS", false, true),
      true >> (edm::ParameterDescription<edm::InputTag>("inputPS", edm::InputTag(""), true)) or
          false >> (edm::ParameterDescription<edm::InputTag>("inputPS", edm::InputTag("particleFlowClusterPS"), true)));
  {
    edm::ParameterSetDescription psd0;
    psd0.add<bool>("applyCrackCorrections", false);
    psd0.add<bool>("applyMVACorrections", false);
    psd0.add<bool>("srfAwareCorrection", false);
    psd0.add<bool>("setEnergyUncertainty", false);
    psd0.add<bool>("autoDetectBunchSpacing", true);
    psd0.add<int>("bunchSpacing", 25);
    psd0.add<double>("maxPtForMVAEvaluation", -99.);
    psd0.add<edm::InputTag>("recHitsEBLabel", edm::InputTag("ecalRecHit", "EcalRecHitsEB"));
    psd0.add<edm::InputTag>("recHitsEELabel", edm::InputTag("ecalRecHit", "EcalRecHitsEE"));
    psd0.add<edm::InputTag>("ebSrFlagLabel", edm::InputTag("ecalDigis"));
    psd0.add<edm::InputTag>("eeSrFlagLabel", edm::InputTag("ecalDigis"));
    desc.add<edm::ParameterSetDescription>("energyCorrector", psd0);
  }
  desc.add<edm::InputTag>("inputECAL", edm::InputTag("particleFlowClusterECALUncorrected"));
  descriptions.add("particleFlowClusterECAL", desc);
}

#endif
