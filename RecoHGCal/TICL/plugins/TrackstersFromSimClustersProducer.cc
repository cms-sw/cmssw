// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 02/2021

// user include files
#include <vector>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "SimDataFormats/Associations/interface/LayerClusterToSimClusterAssociator.h"

using namespace ticl;

class TrackstersFromSimClustersProducer : public edm::stream::EDProducer<> {
public:
  explicit TrackstersFromSimClustersProducer(const edm::ParameterSet&);
  ~TrackstersFromSimClustersProducer() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event&, const edm::EventSetup&) override;


private:
  std::string detector_;
  bool doNose_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> clusters_token_;
  const edm::EDGetTokenT<edm::ValueMap<std::pair<float, float>>> clustersTime_token_;
  edm::EDGetTokenT<TICLLayerTiles> layer_clusters_tiles_token_;
  edm::EDGetTokenT<TICLLayerTilesHFNose> layer_clusters_tiles_hfnose_token_;
  edm::EDGetTokenT<std::vector<SimCluster>> simclusters_token_;

  edm::InputTag associatorLayerClusterSimCluster_;
  edm::EDGetTokenT<hgcal::SimToRecoCollectionWithSimClusters> associatorMapSimToReco_token_;


};
DEFINE_FWK_MODULE(TrackstersFromSimClustersProducer);



TrackstersFromSimClustersProducer::TrackstersFromSimClustersProducer(const edm::ParameterSet& ps)
    : detector_(ps.getParameter<std::string>("detector")),
      doNose_(detector_ == "HFNose"),
      clusters_token_(consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layer_clusters"))),
      clustersTime_token_(consumes<edm::ValueMap<std::pair<float, float>>>(ps.getParameter<edm::InputTag>("time_layerclusters"))),
      simclusters_token_(consumes<std::vector<SimCluster>>(ps.getParameter<edm::InputTag>("simclusters"))),
      associatorLayerClusterSimCluster_(ps.getUntrackedParameter<edm::InputTag>("layerClusterSimClusterAssociator")),
      associatorMapSimToReco_token_(consumes<hgcal::SimToRecoCollectionWithSimClusters>(associatorLayerClusterSimCluster_))
{

  if (doNose_) {
    layer_clusters_tiles_hfnose_token_ =
        consumes<TICLLayerTilesHFNose>(ps.getParameter<edm::InputTag>("layer_clusters_hfnose_tiles"));
  } else {
    layer_clusters_tiles_token_ = consumes<TICLLayerTiles>(ps.getParameter<edm::InputTag>("layer_clusters_tiles"));
  }
  produces<std::vector<Trackster>>();
}

void TrackstersFromSimClustersProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // hgcalMultiClusters
  edm::ParameterSetDescription desc;
  desc.add<std::string>("detector", "HGCAL");
  desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hgcalLayerClusters"));
  desc.add<edm::InputTag>("time_layerclusters", edm::InputTag("hgcalLayerClusters", "timeLayerCluster"));
  desc.add<edm::InputTag>("layer_clusters_tiles", edm::InputTag("ticlLayerTileProducer"));
  desc.add<edm::InputTag>("layer_clusters_hfnose_tiles", edm::InputTag("ticlLayerTileHFNose"));
  desc.add<edm::InputTag>("simclusters", edm::InputTag("mix","MergedCaloTruth"));
  desc.addUntracked<edm::InputTag>("layerClusterSimClusterAssociator", edm::InputTag("layerClusterSimClusterAssociationProducer"));
  descriptions.add("trackstersFromSimClustersProducer", desc);
}

void TrackstersFromSimClustersProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  auto result = std::make_unique<std::vector<Trackster>>();
  const auto& layerClusters = evt.get(clusters_token_);
  const auto& layerClustersTimes = evt.get(clustersTime_token_);
  const auto& simclusters = evt.get(simclusters_token_);
  const auto& simToRecoColl = evt.get(associatorMapSimToReco_token_);
  std::cout << "pippo" << std::endl;

  
  // const auto& lcsIt = simToRecoColl.find(scRef);

  

  evt.put(std::move(result));
}
