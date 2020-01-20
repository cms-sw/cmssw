#ifndef __RecoLocalCalo_HGCRecProducers_HGCalLayerClusterProducer_H__
#define __RecoLocalCalo_HGCRecProducers_HGCalLayerClusterProducer_H__

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/PluginDescription.h"

#include "RecoParticleFlow/PFClusterProducer/interface/RecHitTopologicalCleanerBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/SeedFinderBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterBuilderBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterEnergyCorrectorBase.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/ComputeClusterTime.h"

#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalLayerClusterAlgoFactory.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalDepthPreClusterer.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/Common/interface/ValueMap.h"

using Density = hgcal_clustering::Density;

class HGCalLayerClusterProducer : public edm::stream::EDProducer<> {
public:
  HGCalLayerClusterProducer(const edm::ParameterSet&);
  ~HGCalLayerClusterProducer() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<HGCRecHitCollection> hits_ee_token;
  edm::EDGetTokenT<HGCRecHitCollection> hits_fh_token;
  edm::EDGetTokenT<HGCRecHitCollection> hits_bh_token;
  edm::EDGetTokenT<HGCRecHitCollection> hits_hfnose_token;

  reco::CaloCluster::AlgoId algoId;

  std::unique_ptr<HGCalClusteringAlgoBase> algo;
  bool doSharing;
  std::string detector;

  std::string timeClname;
  double timeOffset;
  unsigned int nHitsTime;
};

DEFINE_FWK_MODULE(HGCalLayerClusterProducer);

HGCalLayerClusterProducer::HGCalLayerClusterProducer(const edm::ParameterSet& ps)
    : algoId(reco::CaloCluster::undefined),
      doSharing(ps.getParameter<bool>("doSharing")),
      detector(ps.getParameter<std::string>("detector")),  // one of EE, FH, BH or "all"
      timeClname(ps.getParameter<std::string>("timeClname")),
      timeOffset(ps.getParameter<double>("timeOffset")),
      nHitsTime(ps.getParameter<unsigned int>("nHitsTime")) {
  if (detector == "HFNose") {
    hits_hfnose_token = consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("HFNoseInput"));
    algoId = reco::CaloCluster::hfnose;
  } else if (detector == "all") {
    hits_ee_token = consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("HGCEEInput"));
    hits_fh_token = consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("HGCFHInput"));
    hits_bh_token = consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("HGCBHInput"));
    algoId = reco::CaloCluster::hgcal_mixed;
  } else if (detector == "EE") {
    hits_ee_token = consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("HGCEEInput"));
    algoId = reco::CaloCluster::hgcal_em;
  } else if (detector == "FH") {
    hits_fh_token = consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("HGCFHInput"));
    algoId = reco::CaloCluster::hgcal_had;
  } else {
    hits_bh_token = consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("HGCBHInput"));
    algoId = reco::CaloCluster::hgcal_had;
  }

  auto pluginPSet = ps.getParameter<edm::ParameterSet>("plugin");
  if (detector == "HFNose") {
    algo = HGCalLayerClusterAlgoFactory::get()->create("HFNoseCLUE", pluginPSet);
  } else {
    algo = HGCalLayerClusterAlgoFactory::get()->create(pluginPSet.getParameter<std::string>("type"), pluginPSet);
  }

  algo->setAlgoId(algoId);

  produces<std::vector<float>>("InitialLayerClustersMask");
  produces<std::vector<reco::BasicCluster>>();
  produces<std::vector<reco::BasicCluster>>("sharing");
  //density
  produces<Density>();
  //time for layer clusters
  produces<edm::ValueMap<std::pair<float, float>>>(timeClname);
}

void HGCalLayerClusterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // hgcalLayerClusters
  edm::ParameterSetDescription desc;
  edm::ParameterSetDescription pluginDesc;
  pluginDesc.addNode(edm::PluginDescription<HGCalLayerClusterAlgoFactory>("type", "CLUE", true));

  desc.add<edm::ParameterSetDescription>("plugin", pluginDesc);
  desc.add<std::string>("detector", "all")
      ->setComment("all (does not include HFNose); other options: EE, FH, HFNose; other value defaults to BH");
  desc.add<bool>("doSharing", false);
  desc.add<edm::InputTag>("HFNoseInput", edm::InputTag("HGCalRecHit", "HGCHFNoseRecHits"));
  desc.add<edm::InputTag>("HGCEEInput", edm::InputTag("HGCalRecHit", "HGCEERecHits"));
  desc.add<edm::InputTag>("HGCFHInput", edm::InputTag("HGCalRecHit", "HGCHEFRecHits"));
  desc.add<edm::InputTag>("HGCBHInput", edm::InputTag("HGCalRecHit", "HGCHEBRecHits"));
  desc.add<std::string>("timeClname", "timeLayerCluster");
  desc.add<double>("timeOffset", 0.0);
  desc.add<unsigned int>("nHitsTime", 3);
  descriptions.add("hgcalLayerClusters", desc);
}

void HGCalLayerClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  edm::Handle<HGCRecHitCollection> hfnose_hits;
  edm::Handle<HGCRecHitCollection> ee_hits;
  edm::Handle<HGCRecHitCollection> fh_hits;
  edm::Handle<HGCRecHitCollection> bh_hits;

  std::unique_ptr<std::vector<reco::BasicCluster>> clusters(new std::vector<reco::BasicCluster>),
      clusters_sharing(new std::vector<reco::BasicCluster>);
  auto density = std::make_unique<Density>();

  algo->reset();

  algo->getEventSetup(es);

  //make a map detid-rechit
  // NB for the moment just host EE and FH hits
  // timing in digi for BH not implemented for now
  std::unordered_map<uint32_t, const HGCRecHit*> hitmap;

  switch (algoId) {
    case reco::CaloCluster::hfnose:
      evt.getByToken(hits_hfnose_token, hfnose_hits);
      algo->populate(*hfnose_hits);
      for (auto const& it : *hfnose_hits)
        hitmap[it.detid().rawId()] = &(it);
      break;
    case reco::CaloCluster::hgcal_em:
      evt.getByToken(hits_ee_token, ee_hits);
      algo->populate(*ee_hits);
      for (auto const& it : *ee_hits)
        hitmap[it.detid().rawId()] = &(it);
      break;
    case reco::CaloCluster::hgcal_had:
      evt.getByToken(hits_fh_token, fh_hits);
      evt.getByToken(hits_bh_token, bh_hits);
      if (fh_hits.isValid()) {
        algo->populate(*fh_hits);
        for (auto const& it : *fh_hits)
          hitmap[it.detid().rawId()] = &(it);
      } else if (bh_hits.isValid()) {
        algo->populate(*bh_hits);
      }
      break;
    case reco::CaloCluster::hgcal_mixed:
      evt.getByToken(hits_ee_token, ee_hits);
      algo->populate(*ee_hits);
      for (auto const& it : *ee_hits) {
        hitmap[it.detid().rawId()] = &(it);
      }
      evt.getByToken(hits_fh_token, fh_hits);
      algo->populate(*fh_hits);
      for (auto const& it : *fh_hits) {
        hitmap[it.detid().rawId()] = &(it);
      }
      evt.getByToken(hits_bh_token, bh_hits);
      algo->populate(*bh_hits);
      break;
    default:
      break;
  }
  algo->makeClusters();
  *clusters = algo->getClusters(false);
  if (doSharing)
    *clusters_sharing = algo->getClusters(true);

  auto clusterHandle = evt.put(std::move(clusters));
  auto clusterHandleSharing = evt.put(std::move(clusters_sharing), "sharing");

  //Keep the density
  *density = algo->getDensity();
  evt.put(std::move(density));

  edm::PtrVector<reco::BasicCluster> clusterPtrs, clusterPtrsSharing;

  std::vector<std::pair<float, float>> times;
  times.reserve(clusterHandle->size());

  for (unsigned i = 0; i < clusterHandle->size(); ++i) {
    edm::Ptr<reco::BasicCluster> ptr(clusterHandle, i);
    clusterPtrs.push_back(ptr);

    std::pair<float, float> timeCl(-99., -1.);

    const reco::CaloCluster& sCl = (*clusterHandle)[i];
    if (sCl.size() >= nHitsTime) {
      std::vector<float> timeClhits;
      std::vector<float> timeErrorClhits;

      for (auto const& hit : sCl.hitsAndFractions()) {
        auto finder = hitmap.find(hit.first);
        if (finder == hitmap.end())
          continue;

        //time is computed wrt  0-25ns + offset and set to -1 if no time
        const HGCRecHit* rechit = finder->second;
        float rhTimeE = rechit->timeError();
        //check on timeError to exclude scintillator
        if (rhTimeE < 0.)
          continue;
        timeClhits.push_back(rechit->time() - timeOffset);
        timeErrorClhits.push_back(1. / (rhTimeE * rhTimeE));
      }
      hgcalsimclustertime::ComputeClusterTime timeEstimator;
      timeCl = timeEstimator.fixSizeHighestDensity(timeClhits, timeErrorClhits, nHitsTime);
    }
    times.push_back(timeCl);
  }
  std::unique_ptr<std::vector<float>> layerClustersMask(new std::vector<float>);
  layerClustersMask->resize(clusterHandle->size(), 1.0);
  evt.put(std::move(layerClustersMask), "InitialLayerClustersMask");

  auto timeCl = std::make_unique<edm::ValueMap<std::pair<float, float>>>();
  edm::ValueMap<std::pair<float, float>>::Filler filler(*timeCl);
  filler.insert(clusterHandle, times.begin(), times.end());
  filler.fill();
  evt.put(std::move(timeCl), timeClname);

  if (doSharing) {
    for (unsigned i = 0; i < clusterHandleSharing->size(); ++i) {
      edm::Ptr<reco::BasicCluster> ptr(clusterHandleSharing, i);
      clusterPtrsSharing.push_back(ptr);
    }
  }
}

#endif
