// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 09/2018
// Copyright CERN

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ESHandle.h"


#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "FilteredLayerClustersProducer.h"
#include "RecoHGCal/TICL/interface/ClusterFilter.h"


DEFINE_FWK_MODULE(FilteredLayerClustersProducer);

FilteredLayerClustersProducer::FilteredLayerClustersProducer(
    const edm::ParameterSet& ps)
    {
  
  clusters_token = consumes<std::vector<reco::CaloCluster>>(
      ps.getParameter<edm::InputTag>("HGCLayerClusters"));
  clustersMask_token = consumes<std::vector<float>>(
      ps.getParameter<edm::InputTag>("LayerClustersInputMask"));
  edm::InputTag clusterFilterTag =
      ps.getParameter<edm::InputTag>("ClusterFilter");
  if (clusterFilterTag.label() != "") {
    clusterFilterToken = consumes<ClusterFilter>(clusterFilterTag);
  }
  produces<std::vector<std::pair<unsigned int, float>>>("iterationLabelGoesHere");
}

void FilteredLayerClustersProducer::fillDescriptions(
    edm::ConfigurationDescriptions& descriptions) {
  // hgcalMultiClusters
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("HGCLayerClusters",
                          edm::InputTag("hgcalLayerClusters"));
  desc.add<edm::InputTag>(
      "LayerClustersInputMask",
      edm::InputTag("hgcalLayerClusters", "InitialLayerClustersMask"));
  desc.add<edm::InputTag>(
      "ClusterFilter",
      edm::InputTag(""));
  descriptions.add("FilteredLayerClusters", desc);
}



void FilteredLayerClustersProducer::produce(edm::Event& evt,
                                          const edm::EventSetup& es) {
  edm::Handle<std::vector<reco::CaloCluster>> clusterHandle;
  edm::Handle<std::vector<float>> inputClustersMaskHandle;
  auto maskedLayerClusters = std::make_unique<std::vector<std::pair<unsigned int, float>>>();
  evt.getByToken(clusters_token, clusterHandle);
  evt.getByToken(clustersMask_token, inputClustersMaskHandle);
  const auto& inputClusterMask = *inputClustersMaskHandle;

  const auto& layerClusters = *clusterHandle;
  auto numLayerClusters = layerClusters.size();
  maskedLayerClusters->reserve(numLayerClusters);
  for (unsigned int i = 0; i < numLayerClusters; ++i) {
    if (inputClusterMask[i] > 0.f) {
      maskedLayerClusters->emplace_back(std::make_pair(i, inputClusterMask[i]));
    }
  }

  if (!clusterFilterToken.isUninitialized()) {
    edm::Handle<ClusterFilter> hfilter;
    evt.getByToken(clusterFilterToken, hfilter);
    theFilter = hfilter.product();
  }

  std::unique_ptr<std::vector<std::pair<unsigned int, float>>> filteredLayerClusters;
  if (theFilter) {
    filteredLayerClusters= theFilter->filter(layerClusters, *maskedLayerClusters);
  }
  else
  {
    filteredLayerClusters = std::move(maskedLayerClusters);
  }
  evt.put(std::move(filteredLayerClusters), "iterationLabelGoesHere");
}
