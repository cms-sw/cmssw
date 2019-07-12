// Author: Felice Pantaleo, Marco Rovere - felice.pantaleo@cern.ch, marco.rovere@cern.ch
// Date: 11/2018
#include <algorithm>
#include <set>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "PatternRecognitionbyCA.h"
#include "HGCGraph.h"

using namespace ticl;

PatternRecognitionbyCA::PatternRecognitionbyCA(const edm::ParameterSet &conf,
    tf::GraphDef* energyIDGraphDef)
    : PatternRecognitionAlgoBase(conf), energyIDSession_(nullptr) {
  theGraph_ = std::make_unique<HGCGraph>();
  min_cos_theta_ = conf.getParameter<double>("min_cos_theta");
  min_cos_pointing_ = conf.getParameter<double>("min_cos_pointing");
  missing_layers_ = conf.getParameter<int>("missing_layers");
  min_clusters_per_ntuplet_ = conf.getParameter<int>("min_clusters_per_ntuplet");
  max_delta_time_ = conf.getParameter<double>("max_delta_time");

  // mount the tensorflow graph onto the session when set
  if (energyIDGraphDef != nullptr) {
    energyIDSession_ = tf::createSession(energyIDGraphDef);
  }
}

PatternRecognitionbyCA::~PatternRecognitionbyCA(){};

void PatternRecognitionbyCA::makeTracksters(const edm::Event &ev,
                                            const edm::EventSetup &es,
                                            const std::vector<reco::CaloCluster> &layerClusters,
                                            const std::vector<float> &mask,
                                            const edm::ValueMap<float> &layerClustersTime,
                                            const TICLLayerTiles &tiles,
                                            std::vector<Trackster> &result) {
  rhtools_.getEventSetup(es);

  theGraph_->setVerbosity(algo_verbosity_);
  theGraph_->clear();
  if (algo_verbosity_ > None) {
    LogDebug("HGCPatterRecoByCA") << "Making Tracksters with CA" << std::endl;
  }
  std::vector<HGCDoublet::HGCntuplet> foundNtuplets;
  std::vector<uint8_t> layer_cluster_usage(layerClusters.size(), 0);
  theGraph_->makeAndConnectDoublets(tiles,
                                    ticl::constants::nEtaBins,
                                    ticl::constants::nPhiBins,
                                    layerClusters,
                                    mask,
                                    layerClustersTime,
                                    2,
                                    2,
                                    min_cos_theta_,
                                    min_cos_pointing_,
                                    missing_layers_,
                                    rhtools_.lastLayerFH(),
                                    max_delta_time_);
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
        LogDebug("HGCPatterRecoByCA") << "New doublet " << doublet << " for trackster: " << result.size() << " InnerCl "
                                      << innerCluster << " " << layerClusters[innerCluster].x() << " "
                                      << layerClusters[innerCluster].y() << " " << layerClusters[innerCluster].z()
                                      << " OuterCl " << outerCluster << " " << layerClusters[outerCluster].x() << " "
                                      << layerClusters[outerCluster].y() << " " << layerClusters[outerCluster].z()
                                      << " " << tracksterId << std::endl;
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

  // energy regression and ID when a session is created
  if (energyIDSession_ != nullptr) {
    energyRegressionAndID(layerClusters, result);
  }
}

void PatternRecognitionbyCA::energyRegressionAndID(
    const std::vector<reco::CaloCluster>& layerClusters, std::vector<Trackster>& result) {
  // TODO: the inference doesn't use batching yet!
  // TODO: layer clusters are not sorted by any metric yet, some geometric approach might be good,
  // maybe even correlated between layers

  // define input dimensions
  // TODO: use instance members
  size_t batchSize = 1;
  size_t nLayers = 50;
  size_t nClusters = 10;
  size_t nFeatures = 3;
  float zero = 0.;

  // create structures for input / output tensors as shapes are distinct as long as we do not batch
  // TODO: tensor names might become configurable via pset
  std::vector<tf::Tensor> outputs;
  tf::Tensor input(tf::DT_FLOAT,
    tf::TensorShape({ (int)batchSize, (int)nLayers, (int)nClusters, (int)nFeatures }));
  tf::NamedTensorList inputList = { { "input", input } };
  std::vector<std::string> outputNames = { "output/Softmax" };

  for (size_t i = 0; i < result.size(); i++) {
    // only run the inference for tracksters whose sum of layer cluster energies is > 5 GeV
    // note: after the loop, sumClusterEnergy might be just above 5, which is enough to decide
    // whether to run inference for the trackster or not
    float sumClusterEnergy = 0.;
    for(size_t cluster = 0; cluster < result[i].vertices.size(); cluster++) {
      sumClusterEnergy += (float)layerClusters[result[i].vertices[cluster]].energy();
      // there might be many clusters, so try to stop early
      if (sumClusterEnergy > 5) {
        break;
      }
    }
    if (sumClusterEnergy <= 5) {
      // TODO: set probabilities and regressed energy to default values? if so, this should maybe happen via
      // a method of the trackster struct itself
      continue;
    }

    // get features per layer and cluster and store them in a nested vector
    // this is necessary since layer information is stored per layer cluster, not vice versa
    std::vector<std::vector<std::array<float, 3> > > tracksterFeatures;
    tracksterFeatures.resize(nLayers);
    for(size_t cluster = 0; cluster < result[i].vertices.size(); cluster++) {
      const reco::CaloCluster& lc = layerClusters[result[i].vertices[cluster]];
      // TODO: can getLayerWithOffset(lc.hitsAndFractions()[0].first) return 0?
      size_t layer = rhtools_.getLayerWithOffset(lc.hitsAndFractions()[0].first) - 1;
      if (layer < nLayers) {
        std::array<float, 3> features {{ float(lc.eta()), float(lc.phi()), float(lc.energy()) }};
        tracksterFeatures[layer].push_back(features);
      }
    }

    // TODO: sorting of layer clusters (even correlated between layers) could happen here

    // start filling input tensor data
    float* data = input.flat<float>().data();
    for (size_t layer = 0; layer < nLayers; layer++) {
      for (size_t cluster = 0; cluster < nClusters; cluster++) {
        // if there are not enough clusters, fill zeros
        if (cluster < tracksterFeatures[layer].size()) {
          std::array<float, 3>& features = tracksterFeatures[layer][cluster];
          *(data++) = features[0];
          *(data++) = features[1];
          *(data++) = features[2];
        } else {
          *(data++) = zero;
          *(data++) = zero;
          *(data++) = zero;
        }
      }
    }

    // run the inference
    tf::run(energyIDSession_, inputList, outputNames, &outputs);

    // store ID probabilities in trackster
    float* probs = outputs[0].flat<float>().data();
    result[i].prob_photon = *(probs++);
    result[i].prob_electron = *(probs++);
    result[i].prob_muon = *(probs++);
    result[i].prob_charged_pion = *(probs++);

    // TODO: store regressed energy when the model is capable of producing that
    result[i].regressed_energy = -1.;

    // some debug log, to be removed for actual PR
    if (i == 0) {
        std::cout << "probabilities of first trackster:" << std::endl;
        std::cout << "photon:   " << result[i].prob_photon << std::endl;
        std::cout << "electron: " << result[i].prob_electron << std::endl;
        std::cout << "muon:     " << result[i].prob_muon << std::endl;
        std::cout << "ch. pion: " << result[i].prob_charged_pion << std::endl;
    }
  }
}
