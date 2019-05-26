// Author: Felice Pantaleo, Marco Rovere - felice.pantaleo@cern.ch, marco.rovere@cern.ch
// Date: 09/2018

// user include files
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "ClusterFilterFactory.h"
#include "ClusterFilterBase.h"

#include <string>

class FilteredLayerClustersProducer : public edm::stream::EDProducer<> {
 public:
  FilteredLayerClustersProducer(const edm::ParameterSet &);
  ~FilteredLayerClustersProducer() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void produce(edm::Event &, const edm::EventSetup &) override;

 private:
  edm::EDGetTokenT<std::vector<reco::CaloCluster>> clusters_token_;
  edm::EDGetTokenT<std::vector<float>> clustersMask_token_;
  std::string clusterFilter_;
  std::string iteration_label_;
  const ticl::ClusterFilterBase *theFilter_ = nullptr;
};

DEFINE_FWK_MODULE(FilteredLayerClustersProducer);

FilteredLayerClustersProducer::FilteredLayerClustersProducer(const edm::ParameterSet& ps) {
  clusters_token_ =
      consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("HGCLayerClusters"));
  clustersMask_token_ =
      consumes<std::vector<float>>(ps.getParameter<edm::InputTag>("LayerClustersInputMask"));
  clusterFilter_ = ps.getParameter<std::string>("clusterFilter");
  theFilter_ = ClusterFilterFactory::get()->create(clusterFilter_, ps);
  iteration_label_ = ps.getParameter<std::string>("iteration_label");

  produces<ticl::HgcalClusterFilterMask>(iteration_label_);
}

void FilteredLayerClustersProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // hgcalMultiClusters
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("HGCLayerClusters", edm::InputTag("hgcalLayerClusters"));
  desc.add<edm::InputTag>("LayerClustersInputMask",
                          edm::InputTag("hgcalLayerClusters", "InitialLayerClustersMask"));
  desc.add<std::string>("iteration_label", "iterationLabelGoesHere");
  desc.add<std::string>("clusterFilter", "ClusterFilterByAlgo");
  desc.add<int>("algo_number", 9);
  desc.add<int>("max_cluster_size", 9999);
  descriptions.add("filteredLayerClustersProducer", desc);
}

void FilteredLayerClustersProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  edm::Handle<std::vector<reco::CaloCluster>> clusterHandle;
  edm::Handle<std::vector<float>> inputClustersMaskHandle;
  auto availableLayerClusters = std::make_unique<ticl::HgcalClusterFilterMask>();
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

  std::unique_ptr<ticl::HgcalClusterFilterMask> filteredLayerClusters;
  if (theFilter_) {
    filteredLayerClusters = theFilter_->filter(layerClusters, *availableLayerClusters);
  } else {
    filteredLayerClusters = std::move(availableLayerClusters);
  }
  evt.put(std::move(filteredLayerClusters), iteration_label_);
}
