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
#include "RecoHGCal/TICL/interface/ClusterFilterFactory.h"


DEFINE_FWK_MODULE(FilteredLayerClustersProducer);

FilteredLayerClustersProducer::FilteredLayerClustersProducer(
    const edm::ParameterSet& ps)
{
  clusters_token_ = consumes<std::vector<reco::CaloCluster>>(
      ps.getParameter<edm::InputTag>("HGCLayerClusters"));
  clustersMask_token_ = consumes<std::vector<float>>(
      ps.getParameter<edm::InputTag>("LayerClustersInputMask"));
  clusterFilter_ =
    ps.getParameter<std::string>("ClusterFilter");
  theFilter_ = ClusterFilterFactory::get()->create(clusterFilter_, ps);

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
  desc.add<std::string>(
      "ClusterFilter", "ClusterFilterByAlgo");
  descriptions.add("FilteredLayerClusters", desc);
}



void FilteredLayerClustersProducer::produce(edm::Event& evt,
    const edm::EventSetup& es) {
  edm::Handle<std::vector<reco::CaloCluster>> clusterHandle;
  edm::Handle<std::vector<float>> inputClustersMaskHandle;
  auto availableLayerClusters = std::make_unique<std::vector<std::pair<unsigned int, float>>>();
  evt.getByToken(clusters_token_, clusterHandle);
  evt.getByToken(clustersMask_token_, inputClustersMaskHandle);
  const auto& inputClusterMask = *inputClustersMaskHandle;

  const auto& layerClusters = *clusterHandle;
  auto numLayerClusters = layerClusters.size();
  availableLayerClusters->reserve(numLayerClusters);
  for (unsigned int i = 0; i < numLayerClusters; ++i) {
    if (inputClusterMask[i] > 0.f) {
      availableLayerClusters->emplace_back(std::make_pair(i, inputClusterMask[i]));
    }
  }

  std::unique_ptr<std::vector<std::pair<unsigned int, float>>> filteredLayerClusters;
  if (theFilter_) {
    filteredLayerClusters = theFilter_->filter(layerClusters, *availableLayerClusters);
  }
  else
  {
    filteredLayerClusters = std::move(availableLayerClusters);
  }
  evt.put(std::move(filteredLayerClusters), "iterationLabelGoesHere");
}
