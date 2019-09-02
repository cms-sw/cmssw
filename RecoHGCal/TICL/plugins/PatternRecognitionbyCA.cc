// Author: Felice Pantaleo, Marco Rovere - felice.pantaleo@cern.ch, marco.rovere@cern.ch
// Date: 11/2018
#include <algorithm>
#include <set>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "PatternRecognitionbyCA.h"
#include "HGCGraph.h"

using namespace ticl;

PatternRecognitionbyCA::PatternRecognitionbyCA(const edm::ParameterSet &conf, const CacheBase *cache)
    : PatternRecognitionAlgoBase(conf, cache), eidSession_(nullptr) {
  theGraph_ = std::make_unique<HGCGraph>();
  min_cos_theta_ = conf.getParameter<double>("min_cos_theta");
  min_cos_pointing_ = conf.getParameter<double>("min_cos_pointing");
  missing_layers_ = conf.getParameter<int>("missing_layers");
  min_clusters_per_ntuplet_ = conf.getParameter<int>("min_clusters_per_ntuplet");
  max_delta_time_ = conf.getParameter<double>("max_delta_time");
  eidInputName_ = conf.getParameter<std::string>("eid_input_name");
  eidOutputNameEnergy_ = conf.getParameter<std::string>("eid_output_name_energy");
  eidOutputNameId_ = conf.getParameter<std::string>("eid_output_name_id");
  eidMinClusterEnergy_ = conf.getParameter<double>("eid_min_cluster_energy");
  eidNLayers_ = conf.getParameter<int>("eid_n_layers");
  eidNClusters_ = conf.getParameter<int>("eid_n_clusters");

  // mount the tensorflow graph onto the session when set
  const TrackstersCache *trackstersCache = dynamic_cast<const TrackstersCache *>(cache);
  if (trackstersCache->eidGraphDef != nullptr) {
    eidSession_ = tensorflow::createSession(trackstersCache->eidGraphDef);
  }
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

  theGraph_->findNtuplets(foundNtuplets, seedIndices, min_clusters_per_ntuplet_);
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

  // energy regression and ID when a session is created
  if (eidSession_ != nullptr) {
    energyRegressionAndID(input.layerClusters, result);
  }
}

void PatternRecognitionbyCA::energyRegressionAndID(const std::vector<reco::CaloCluster> &layerClusters,
                                                   std::vector<Trackster> &tracksters) {
  // Energy regression and particle identification strategy:
  // 1. Set default values for regressed energy and particle id for each trackster.
  // 2. Store indices of tracksters whose total sum of cluster energies is above the
  //    eidMinClusterEnergy_ (GeV) treshold. Inference is not applied for soft tracksters.
  // 3. Create input and output tensors. The batch dimension is determined by the number of
  //    tracksters passing 2.
  // 4. Fill input tensors with variables of tracksters. Per layer, tracksters are ordered
  //    descending by energy. Given that tensor data is contiguous in memory, we can use pointer
  //    arithmetic to fill values, even with batching.
  // 5. Batched inference.
  // 6. Assign regressed energy and id probabilities to each trackster.

  // set default values per trackster, determine if the cluster energy threshold is passed,
  // and store indices of hard tracksters
  std::vector<int> tracksterIndices;
  for (int i = 0; i < (int)tracksters.size(); i++) {
    // set default values (1)
    tracksters[i].regressed_energy = 0.;
    for (size_t j = 0; j < tracksters[i].id_probabilities.size(); j++) {
      tracksters[i].id_probabilities[j] = 0.;
    }

    // calculate the cluster energy sum (2)
    // note: after the loop, sumClusterEnergy might be just above the treshold which is enough to
    // decide whether to run inference for the trackster or not
    float sumClusterEnergy = 0.;
    for (size_t cluster = 0; cluster < tracksters[i].vertices.size(); cluster++) {
      sumClusterEnergy += (float)layerClusters[tracksters[i].vertices[cluster]].energy();
      // there might be many clusters, so try to stop early
      if (sumClusterEnergy >= eidMinClusterEnergy_) {
        tracksterIndices.push_back(i);
        break;
      }
    }
  }

  // create input and output tensors (3)
  int batchSize = (int)tracksterIndices.size();

  tensorflow::TensorShape shape({batchSize, eidNLayers_, eidNClusters_, eidNFeatures_});
  tensorflow::Tensor input(tensorflow::DT_FLOAT, shape);
  tensorflow::NamedTensorList inputList = {{eidInputName_, input}};
  float *inputData = input.flat<float>().data();

  std::vector<tensorflow::Tensor> outputs;
  std::vector<std::string> outputNames;
  if (!eidOutputNameEnergy_.empty()) {
    outputNames.push_back(eidOutputNameEnergy_);
  }
  if (!eidOutputNameId_.empty()) {
    outputNames.push_back(eidOutputNameId_);
  }

  // fill input tensor (4)
  for (int i : tracksterIndices) {
    // get features per layer and cluster and store them in a nested vector
    // this is necessary since layer information is stored per layer cluster, not vice versa
    // also, this allows for convenient re-ordering
    std::vector<std::vector<std::array<float, eidNFeatures_> > > tracksterFeatures;
    tracksterFeatures.resize(eidNLayers_);
    for (int cluster = 0; cluster < (int)tracksters[i].vertices.size(); cluster++) {
      const reco::CaloCluster &lc = layerClusters[tracksters[i].vertices[cluster]];
      int layer = rhtools_.getLayerWithOffset(lc.hitsAndFractions()[0].first) - 1;
      if (layer < eidNLayers_) {
        std::array<float, eidNFeatures_> features{{float(lc.eta()), float(lc.phi()), float(lc.energy())}};
        tracksterFeatures[layer].push_back(features);
      }
    }

    // start filling input tensor data
    for (int layer = 0; layer < eidNLayers_; layer++) {
      // per layer, sort tracksters by decreasing energy
      std::vector<std::array<float, eidNFeatures_> > &layerData = tracksterFeatures[layer];
      sort(layerData.begin(),
           layerData.end(),
           [](const std::array<float, eidNFeatures_> &a, const std::array<float, eidNFeatures_> &b) {
             return a[2] > b[2];
           });

      for (int cluster = 0; cluster < eidNClusters_; cluster++) {
        // if there are not enough clusters, fill zeros
        if (cluster < (int)tracksterFeatures[layer].size()) {
          std::array<float, eidNFeatures_> &features = layerData[cluster];
          for (int j = 0; j < eidNFeatures_; j++) {
            *(inputData++) = features[j];
          }
        } else {
          for (int j = 0; j < eidNFeatures_; j++) {
            *(inputData++) = 0.f;
          }
        }
      }
    }
  }

  // run the inference (5)
  tensorflow::run(eidSession_, inputList, outputNames, &outputs);

  // store regressed energy per trackster (6)
  if (!eidOutputNameEnergy_.empty()) {
    // get the pointer to the energy tensor, dimension is batch x 1
    float *energy = outputs[0].flat<float>().data();

    for (int i : tracksterIndices) {
      tracksters[i].regressed_energy = *(energy++);
    }
  }

  // store id probabilities per trackster (6)
  if (!eidOutputNameId_.empty()) {
    // get the pointer to the id probability tensor, dimension is batch x id_probabilities.size()
    size_t probsIdx = eidOutputNameEnergy_.empty() ? 0 : 1;
    float *probs = outputs[probsIdx].flat<float>().data();

    for (int i : tracksterIndices) {
      for (size_t j = 0; j < tracksters[i].id_probabilities.size(); j++) {
        tracksters[i].id_probabilities[j] = *(probs++);
      }
    }
  }
}
