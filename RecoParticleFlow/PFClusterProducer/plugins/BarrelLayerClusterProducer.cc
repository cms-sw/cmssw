#ifndef __RecoParticleFlow_PFClusterProducer_BarrelLayerClusterProducer_H__
#define __RecoParticleFlow_PFClusterProducer_BarrelLayerClusterProducer_H__

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
#include "RecoParticleFlow/PFClusterProducer/interface/CaloRecHitResolutionProvider.h"

#include "RecoLocalCalo/HGCalRecProducers/interface/ComputeClusterTime.h"

#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalLayerClusterAlgoFactory.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalDepthPreClusterer.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include "DataFormats/Common/interface/ValueMap.h"

#include "RecoParticleFlow/PFClusterProducer/plugins/BarrelCLUEAlgo.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/BarrelTilesConstants.h"

using Density = hgcal_clustering::Density;

class BarrelLayerClusterProducer : public edm::stream::EDProducer<> {
public:
  BarrelLayerClusterProducer(const edm::ParameterSet&);
  ~BarrelLayerClusterProducer() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  reco::CaloCluster::AlgoId algoId;
  std::string timeClname;
  unsigned int nHitsTime;
  edm::EDGetTokenT<reco::PFRecHitCollection> ebhits_token_;
  edm::EDGetTokenT<reco::PFRecHitCollection> hbhits_token_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeomToken_;

  edm::EDPutTokenT<std::vector<float>> initialmask_puttoken_;
  edm::EDPutTokenT<std::vector<reco::BasicCluster>> clusters_puttoken_;
  edm::EDPutTokenT<edm::ValueMap<std::pair<float, float>>> times_puttoken_;

  std::unique_ptr<HGCalClusteringAlgoBase> ebalgo, hbalgo, hoalgo;
  hgcal::RecHitTools rhtools_;
  std::unique_ptr<CaloRecHitResolutionProvider> timeResolutionCalc_;
};

DEFINE_FWK_MODULE(BarrelLayerClusterProducer);

BarrelLayerClusterProducer::BarrelLayerClusterProducer(const edm::ParameterSet& ps)
    : algoId(reco::CaloCluster::undefined),
      timeClname(ps.getParameter<std::string>("timeClname")),
      nHitsTime(ps.getParameter<unsigned int>("nHitsTime")),
      ebhits_token_{consumes<reco::PFRecHitCollection>(ps.getParameter<edm::InputTag>("EBInput"))},
      hbhits_token_{consumes<reco::PFRecHitCollection>(ps.getParameter<edm::InputTag>("HBInput"))},
      caloGeomToken_{consumesCollector().esConsumes<CaloGeometry, CaloGeometryRecord>()} {
  auto ebpluginPSet = ps.getParameter<edm::ParameterSet>("ebplugin");
  ebalgo = HGCalLayerClusterAlgoFactory::get()->create(ebpluginPSet.getParameter<std::string>("type"), ebpluginPSet);
  algoId = reco::CaloCluster::hgcal_em;
  ebalgo->setAlgoId(algoId);

  auto hbpluginPSet = ps.getParameter<edm::ParameterSet>("hbplugin");
  hbalgo = HGCalLayerClusterAlgoFactory::get()->create(hbpluginPSet.getParameter<std::string>("type"), hbpluginPSet);
  algoId = reco::CaloCluster::hgcal_had;
  hbalgo->setAlgoId(algoId);

  ebalgo->setThresholds(consumesCollector().esConsumes<EcalPFRecHitThresholds, EcalPFRecHitThresholdsRcd>(),
                        consumesCollector().esConsumes<HcalPFCuts, HcalPFCutsRcd>());
  hbalgo->setThresholds(consumesCollector().esConsumes<EcalPFRecHitThresholds, EcalPFRecHitThresholdsRcd>(),
                        consumesCollector().esConsumes<HcalPFCuts, HcalPFCutsRcd>());

  timeResolutionCalc_ = std::make_unique<CaloRecHitResolutionProvider>(ps.getParameterSet("timeResolutionCalc"));
  initialmask_puttoken_ = produces<std::vector<float>>("InitialLayerClustersMask");
  clusters_puttoken_ = produces<std::vector<reco::BasicCluster>>();
  //density
  //produces<Density>();
  //time for layer clusters
  times_puttoken_ = produces<edm::ValueMap<std::pair<float, float>>>(timeClname);
}

void BarrelLayerClusterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  edm::ParameterSetDescription ebpluginDesc;
  ebpluginDesc.addNode(edm::PluginDescription<HGCalLayerClusterAlgoFactory>("type", "EBCLUE", true));

  edm::ParameterSetDescription hbpluginDesc;
  hbpluginDesc.addNode(edm::PluginDescription<HGCalLayerClusterAlgoFactory>("type", "HBCLUE", true));

  desc.add<edm::ParameterSetDescription>("ebplugin", ebpluginDesc);
  desc.add<edm::ParameterSetDescription>("hbplugin", hbpluginDesc);

  desc.add<edm::InputTag>("EBInput", edm::InputTag("particleFlowRecHitECAL", ""));
  desc.add<edm::InputTag>("HBInput", edm::InputTag("particleFlowRecHitHBHE", ""));

  edm::ParameterSetDescription timeResolutionCalcDesc;
  timeResolutionCalcDesc.addNode(edm::ParameterDescription<double>("noiseTerm", 1.10889, true) and
                                 edm::ParameterDescription<double>("constantTerm", 0.428192, true) and
                                 edm::ParameterDescription<double>("corrTermLowE", 0.0510871, true) and
                                 edm::ParameterDescription<double>("threshLowE", 0.5, true) and
                                 edm::ParameterDescription<double>("constantTermLowE", 0.0, true) and
                                 edm::ParameterDescription<double>("noiseTermLowE", 1.31883, true) and
                                 edm::ParameterDescription<double>("threshHighE", 5.0, true));
  desc.add<edm::ParameterSetDescription>("timeResolutionCalc", timeResolutionCalcDesc);

  desc.add<std::string>("timeClname", "timeLayerCluster");
  desc.add<unsigned int>("nHitsTime", 3);
  descriptions.add("barrelLayerClusters", desc);
}

void BarrelLayerClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  edm::Handle<reco::PFRecHitCollection> ebhits, hbhits, hohits;
  edm::ESHandle<CaloGeometry> geom = es.getHandle(caloGeomToken_);
  rhtools_.setGeometry(*geom);

  //auto density = std::make_unique<Density>();
  ebalgo->getEventSetup(es, rhtools_);

  //make a map detid-rechit
  // NB for the moment just host EE and FH hits
  // timing in digi for BH not implemented for now
  std::unordered_map<uint32_t, const reco::PFRecHit*> hitmap;

  evt.getByToken(ebhits_token_, ebhits);
  ebalgo->populate(*ebhits);
  for (auto& hit : *ebhits) {
    hitmap[hit.detId()] = &(hit);
  }
  ebalgo->makeClusters();
  std::vector<reco::BasicCluster> ebclusters = ebalgo->getClusters(false);

  hbalgo->getEventSetup(es, rhtools_);

  evt.getByToken(hbhits_token_, hbhits);
  hbalgo->populate(*hbhits);
  for (auto& hit : *hbhits) {
    hitmap[hit.detId()] = &(hit);
  }
  hbalgo->makeClusters();
  std::vector<reco::BasicCluster> hbclusters = hbalgo->getClusters(false);

  std::unique_ptr<std::vector<reco::BasicCluster>> clusters(new std::vector<reco::BasicCluster>);
  (*clusters).reserve(hbclusters.size()+ebclusters.size());
  (*clusters).insert((*clusters).end(), ebclusters.begin(), ebclusters.end());
  (*clusters).insert((*clusters).end(), hbclusters.begin(), hbclusters.end());
  
  auto clusterHandle = evt.put(clusters_puttoken_, std::move(clusters));
  //Keep the density
  /**density = algo->getDensity();
  evt.put(std::move(density));*/

  edm::PtrVector<reco::BasicCluster> clusterPtrs;  //, clusterPtrsSharing;
  
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
        const reco::PFRecHit* rechit = finder->second;
        float rhTimeE = timeResolutionCalc_->timeResolution2(rechit->energy());
        //check on timeError to exclude scintillator
        if (rhTimeE < 0.)
          continue;
        timeClhits.push_back(rechit->time());
        timeErrorClhits.push_back(1. / rhTimeE);
      }
      hgcalsimclustertime::ComputeClusterTime timeEstimator;
      timeCl = timeEstimator.fixSizeHighestDensity(timeClhits, timeErrorClhits, nHitsTime);
    }
    times.push_back(timeCl);
  }

  std::unique_ptr<std::vector<float>> layerClustersMask(new std::vector<float>);
  layerClustersMask->resize(clusterHandle->size(), 1.0);
  evt.put(initialmask_puttoken_, std::move(layerClustersMask));
    
  auto timeCl = std::make_unique<edm::ValueMap<std::pair<float, float>>>();
  edm::ValueMap<std::pair<float, float>>::Filler filler(*timeCl);
  filler.insert(clusterHandle, times.begin(), times.end());
  filler.fill();
  evt.put(times_puttoken_, std::move(timeCl));
   
  /*if (doSharing) {
    for (unsigned i = 0; i < clusterHandleSharing->size(); ++i) {
      edm::Ptr<reco::BasicCluster> ptr(clusterHandleSharing, i);
      clusterPtrsSharing.push_back(ptr);
    }
  }*/
  ebalgo->reset();
  hbalgo->reset();
}

#endif  //__RecoLocalCalo_HGCRecProducers_BarrelLayerClusterProducer_H__
