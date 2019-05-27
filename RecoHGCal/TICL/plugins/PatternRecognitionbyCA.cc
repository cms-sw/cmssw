// Author: Felice Pantaleo, Marco Rovere - felice.pantaleo@cern.ch, marco.rovere@cern.ch
// Date: 11/2018
#include <algorithm>
#include <set>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "PatternRecognitionbyCA.h"
#include "HGCGraph.h"

using namespace ticl;

PatternRecognitionbyCA::PatternRecognitionbyCA(const edm::ParameterSet& conf)
  : PatternRecognitionAlgoBase(conf) {
        theGraph_ = std::make_unique<HGCGraph>();
        min_cos_theta_ = (float)conf.getParameter<double>("min_cos_theta");
        min_cos_pointing_ = (float)conf.getParameter<double>("min_cos_pointing");
        missing_layers_ = conf.getParameter<int>("missing_layers");
        min_clusters_per_ntuplet_ = conf.getParameter<int>("min_clusters_per_ntuplet");
      }

PatternRecognitionbyCA::~PatternRecognitionbyCA(){};

void PatternRecognitionbyCA::fillHistogram(
    const std::vector<reco::CaloCluster> &layerClusters,
    const HgcalClusterFilterMask &mask) {
  if (algo_verbosity_ > None) {
    LogDebug("HGCPatterRecoByCA") << "filling eta/phi histogram per Layer" << std::endl;
  }
  for (auto &m : mask) {
    auto lcId = m.first;
    const auto &lc = layerClusters[lcId];
    // getting the layer Id from the detId of the first hit of the layerCluster
    const auto firstHitDetId = lc.hitsAndFractions()[0].first;
    int layer = rhtools_.getLayerWithOffset(firstHitDetId) +
                rhtools_.lastLayerFH() * ((rhtools_.zside(firstHitDetId) + 1) >> 1) - 1;
    assert(layer >= 0);
    auto etaBin = getEtaBin(lc.eta());
    auto phiBin = getPhiBin(lc.phi());
    tile_[layer][globalBin(etaBin, phiBin)].push_back(lcId);
    if (algo_verbosity_ > Advanced) {
      LogDebug("HGCPatterRecoByCA")
          << "Adding layerClusterId: " << lcId << " into bin [eta,phi]: [ " << etaBin << ", "
          << phiBin << "] for layer: " << layer << std::endl;
    }
  }
}

void PatternRecognitionbyCA::makeTracksters(const edm::Event &ev, const edm::EventSetup &es,
                                            const std::vector<reco::CaloCluster> &layerClusters,
                                            const HgcalClusterFilterMask &mask,
                                            std::vector<Trackster> &result) {
  rhtools_.getEventSetup(es);

  clearHistogram();
  theGraph_->setVerbosity(algo_verbosity_);
  theGraph_->clear();
  if (algo_verbosity_ > None) {
    LogDebug("HGCPatterRecoByCA") << "Making Tracksters with CA" << std::endl;
  }
  std::vector<HGCDoublet::HGCntuplet> foundNtuplets;
  fillHistogram(layerClusters, mask);
  std::vector<uint8_t> layer_cluster_usage(layerClusters.size(), 0);
  theGraph_->makeAndConnectDoublets(tile_, patternbyca::nEtaBins, patternbyca::nPhiBins, layerClusters, 2, 2,
                                   min_cos_theta_, min_cos_pointing_, missing_layers_,
                                   rhtools_.lastLayerFH());
  theGraph_->findNtuplets(foundNtuplets, min_clusters_per_ntuplet_);
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
        LogDebug("HGCPatterRecoByCA")
            << "New doublet " << doublet << " for trackster: " << result.size() << " InnerCl "
            << innerCluster << " " << layerClusters[innerCluster].x() << " "
            << layerClusters[innerCluster].y() << " " << layerClusters[innerCluster].z()
            << " OuterCl " << outerCluster << " " << layerClusters[outerCluster].x() << " "
            << layerClusters[outerCluster].y() << " " << layerClusters[outerCluster].z() << " "
            << tracksterId << std::endl;
      }
    }
    for (auto const i : effective_cluster_idx) {
      layer_cluster_usage[i]++;
      LogDebug("HGCPatterRecoByCA")
        << "LayerID: " << i << " count: "
        << (int)layer_cluster_usage[i] << std::endl;
    }
    // Put back indices, in the form of a Trackster, into the results vector
    Trackster tmp;
    tmp.vertices.reserve(effective_cluster_idx.size());
    tmp.vertex_multiplicity.resize(effective_cluster_idx.size(), 0);
    std::copy(std::begin(effective_cluster_idx), std::end(effective_cluster_idx),
              std::back_inserter(tmp.vertices));
    result.push_back(tmp);
    tracksterId++;
  }
  for (auto & trackster : result) {
    for (size_t i = 0; i < trackster.vertices.size(); ++i) {
      assert(i < trackster.vertex_multiplicity.size());
      trackster.vertex_multiplicity[i] = layer_cluster_usage[trackster.vertices[i]];
      LogDebug("HGCPatterRecoByCA") << "LayerIDXX: " << trackster.vertices[i]
        << " count: " << (int)trackster.vertex_multiplicity[i] << std::endl;
    }
  }
  //#endif
}
