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
  reco::CaloCluster::AlgoId algoId_;
  std::string timeClname_;
  unsigned int nHitsTime_;
  edm::EDGetTokenT<reco::PFRecHitCollection> hits_token_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeomToken_;

  std::unique_ptr<HGCalClusteringAlgoBase> algo_;
  hgcal::RecHitTools rhtools_;
  std::unique_ptr<CaloRecHitResolutionProvider> timeResolutionCalc_;

  void setAlgoId(std::string& type);
};

DEFINE_FWK_MODULE(BarrelLayerClusterProducer);

BarrelLayerClusterProducer::BarrelLayerClusterProducer(const edm::ParameterSet& ps)
    : algoId_(reco::CaloCluster::undefined),
      timeClname_(ps.getParameter<std::string>("timeClname")),
      nHitsTime_(ps.getParameter<unsigned int>("nHitsTime")),
      hits_token_{consumes<reco::PFRecHitCollection>(ps.getParameter<edm::InputTag>("recHits"))},
      caloGeomToken_{consumesCollector().esConsumes<CaloGeometry, CaloGeometryRecord>()} {
  auto pluginPSet = ps.getParameter<edm::ParameterSet>("plugin");
  std::string type = pluginPSet.getParameter<std::string>("type");
  algo_ = HGCalLayerClusterAlgoFactory::get()->create(type, pluginPSet);
  setAlgoId(type);

  algo_->setThresholds(consumesCollector().esConsumes<EcalPFRecHitThresholds, EcalPFRecHitThresholdsRcd>(),
                       consumesCollector().esConsumes<HcalPFCuts, HcalPFCutsRcd>());

  timeResolutionCalc_ = std::make_unique<CaloRecHitResolutionProvider>(ps.getParameterSet("timeResolutionCalc"));
  produces<std::vector<float>>("InitialLayerClustersMask");
  produces<std::vector<reco::BasicCluster>>();
  produces<edm::ValueMap<std::pair<float, float>>>(timeClname_);
}

void BarrelLayerClusterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  edm::ParameterSetDescription pluginDesc;
  pluginDesc.addNode(edm::PluginDescription<HGCalLayerClusterAlgoFactory>("type", "EBCLUE", true));

  desc.add<edm::ParameterSetDescription>("plugin", pluginDesc);

  desc.add<edm::InputTag>("recHits", edm::InputTag("particleFlowRecHitECAL", ""));

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
  edm::Handle<reco::PFRecHitCollection> hits;
  edm::ESHandle<CaloGeometry> geom = es.getHandle(caloGeomToken_);
  rhtools_.setGeometry(*geom);

  //auto density = std::make_unique<Density>();
  algo_->getEventSetup(es, rhtools_);

  //make a map detid-rechit
  // NB for the moment just host EE and FH hits
  // timing in digi for BH not implemented for now
  std::unordered_map<uint32_t, const reco::PFRecHit*> hitmap;

  evt.getByToken(hits_token_, hits);
  algo_->populate(*hits);
  for (auto& hit : *hits) {
    hitmap[hit.detId()] = &(hit);
  }
  algo_->makeClusters();

  std::unique_ptr<std::vector<reco::CaloCluster>> clusters(new std::vector<reco::CaloCluster>);
  *clusters = algo_->getClusters(false);
  auto clusterHandle = evt.put(std::move(clusters));

  edm::PtrVector<reco::BasicCluster> clusterPtrs;  //, clusterPtrsSharing;

  std::vector<std::pair<float, float>> times;
  times.reserve(clusterHandle->size());

  for (unsigned i = 0; i < clusterHandle->size(); ++i) {
    edm::Ptr<reco::BasicCluster> ptr(clusterHandle, i);
    clusterPtrs.push_back(ptr);

    std::pair<float, float> timeCl(-99., -1.);

    const reco::CaloCluster& sCl = (*clusterHandle)[i];
    if (sCl.size() >= nHitsTime_) {
      std::vector<float> timeClhits;
      std::vector<float> timeErrorClhits;

      for (auto const& hit : sCl.hitsAndFractions()) {
        auto finder = hitmap.find(hit.first);
        if (finder == hitmap.end())
          continue;

        //time is computed wrt  0-25ns + offset and set to -1 if no time
        const reco::PFRecHit* rechit = finder->second;
        float rhTimeE = timeResolutionCalc_->timeResolution2(rechit->energy());
        if (rhTimeE < 0.)
          continue;
        timeClhits.push_back(rechit->time());
        timeErrorClhits.push_back(1. / rhTimeE);
      }
      hgcalsimclustertime::ComputeClusterTime timeEstimator;
      timeCl = timeEstimator.fixSizeHighestDensity(timeClhits, timeErrorClhits, nHitsTime_);
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
  evt.put(std::move(timeCl), timeClname_);

  algo_->reset();
}

void BarrelLayerClusterProducer::setAlgoId(std::string& type) {
  if (type == "EBCLUE") {
    algoId_ = reco::CaloCluster::barrel_em;
  } else if (type == "HBCLUE") {
    algoId_ = reco::CaloCluster::barrel_had;
  } else {
    throw cms::Exception("InvalidPlugin") << "Invalid plugin type: " << type << std::endl;
  }
}

#endif  //__RecoLocalCalo_HGCRecProducers_BarrelLayerClusterProducer_H__
