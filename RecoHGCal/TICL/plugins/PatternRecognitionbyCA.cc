// Author: Felice Pantaleo, Marco Rovere - felice.pantaleo@cern.ch, marco.rovere@cern.ch
// Date: 11/2018
#include <algorithm>
#include <set>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "PatternRecognitionbyCA.h"
#include "HGCGraph.h"

using namespace ticl;

PatternRecognitionbyCA::PatternRecognitionbyCA(const edm::ParameterSet &conf)
    : PatternRecognitionAlgoBase(conf),
      out_in_dfs_(conf.getParameter<bool>("out_in_dfs")),
      max_out_in_hops_(conf.getParameter<int>("max_out_in_hops")) {
  theGraph_ = std::make_unique<HGCGraph>();
  min_cos_theta_ = conf.getParameter<double>("min_cos_theta");
  min_cos_pointing_ = conf.getParameter<double>("min_cos_pointing");
  missing_layers_ = conf.getParameter<int>("missing_layers");
  min_clusters_per_ntuplet_ = conf.getParameter<int>("min_clusters_per_ntuplet");
  max_delta_time_ = conf.getParameter<double>("max_delta_time");
}

PatternRecognitionbyCA::~PatternRecognitionbyCA(){};

void PatternRecognitionbyCA::makeTracksters(const PatternRecognitionAlgoBase::Inputs &input,
                                            std::vector<Trackster> &result) {
  rhtools_.getEventSetup(input.es);

  theGraph_->setVerbosity(algo_verbosity_);
  theGraph_->clear();
  if (algo_verbosity_ > None) {
    LogDebug("HGCPatterRecoByCA") << "Making Tracksters with CA" << std::endl;
  }
  std::vector<HGCDoublet::HGCntuplet> foundNtuplets;
  std::vector<int> seedIndices;
  std::vector<uint8_t> layer_cluster_usage(input.layerClusters.size(), 0);
  theGraph_->makeAndConnectDoublets(input.tiles,
                                    input.regions,
                                    ticl::constants::nEtaBins,
                                    ticl::constants::nPhiBins,
                                    input.layerClusters,
                                    input.mask,
                                    input.layerClustersTime,
                                    1,
                                    1,
                                    min_cos_theta_,
                                    min_cos_pointing_,
                                    missing_layers_,
                                    rhtools_.lastLayerFH(),
                                    max_delta_time_);

  theGraph_->findNtuplets(foundNtuplets, seedIndices, min_clusters_per_ntuplet_, out_in_dfs_, max_out_in_hops_);
  //#ifdef FP_DEBUG
  const auto &doublets = theGraph_->getAllDoublets();
  int tracksterId = 0;
  for (auto const &ntuplet : foundNtuplets) {
    std::set<unsigned int> effective_cluster_idx;
    for (auto const &doublet : ntuplet) {
      auto innerCluster = doublets[doublet].innerClusterId();
      auto outerCluster = doublets[doublet].outerClusterId();
      effective_cluster_idx.insert(innerCluster);
      effective_cluster_idx.insert(outerCluster);
      if (algo_verbosity_ > Advanced) {
        LogDebug("HGCPatterRecoByCA") << "New doublet " << doublet << " for trackster: " << result.size() << " InnerCl "
                                      << innerCluster << " " << input.layerClusters[innerCluster].x() << " "
                                      << input.layerClusters[innerCluster].y() << " "
                                      << input.layerClusters[innerCluster].z() << " OuterCl " << outerCluster << " "
                                      << input.layerClusters[outerCluster].x() << " "
                                      << input.layerClusters[outerCluster].y() << " "
                                      << input.layerClusters[outerCluster].z() << " " << tracksterId << std::endl;
      }
    }
    for (auto const i : effective_cluster_idx) {
      layer_cluster_usage[i]++;
      LogDebug("HGCPatterRecoByCA") << "LayerID: " << i << " count: " << (int)layer_cluster_usage[i] << std::endl;
    }
    // Put back indices, in the form of a Trackster, into the results vector
    Trackster tmp;
    tmp.vertices.reserve(effective_cluster_idx.size());
    tmp.vertex_multiplicity.resize(effective_cluster_idx.size(), 0);
    //regions and seedIndices can have different size
    //if a seeding region does not lead to any trackster
    tmp.seedID = input.regions[0].collectionID;
    tmp.seedIndex = seedIndices[tracksterId];
    std::copy(std::begin(effective_cluster_idx), std::end(effective_cluster_idx), std::back_inserter(tmp.vertices));
    result.push_back(tmp);
    tracksterId++;
  }

  for (auto &trackster : result) {
    assert(trackster.vertices.size() <= trackster.vertex_multiplicity.size());
    for (size_t i = 0; i < trackster.vertices.size(); ++i) {
      trackster.vertex_multiplicity[i] = layer_cluster_usage[trackster.vertices[i]];
      LogDebug("HGCPatterRecoByCA") << "LayerID: " << trackster.vertices[i]
                                    << " count: " << (int)trackster.vertex_multiplicity[i] << std::endl;
    }
  }
}
