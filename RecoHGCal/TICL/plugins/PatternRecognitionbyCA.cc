// Author: Felice Pantaleo, Marco Rovere - felice.pantaleo@cern.ch, marco.rovere@cern.ch
// Date: 11/2018
#include <algorithm>
#include <set>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "PatternRecognitionbyCA.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/ComputeClusterTime.h"

#include "TrackstersPCA.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/Framework/interface/EventSetup.h"

using namespace ticl;

template <typename TILES>
PatternRecognitionbyCA<TILES>::PatternRecognitionbyCA(const edm::ParameterSet &conf, const CacheBase *cache)
    : PatternRecognitionAlgoBaseT<TILES>(conf, cache),
      theGraph_(std::make_unique<HGCGraphT<TILES>>()),
      oneTracksterPerTrackSeed_(conf.getParameter<bool>("oneTracksterPerTrackSeed")),
      promoteEmptyRegionToTrackster_(conf.getParameter<bool>("promoteEmptyRegionToTrackster")),
      out_in_dfs_(conf.getParameter<bool>("out_in_dfs")),
      max_out_in_hops_(conf.getParameter<int>("max_out_in_hops")),
      min_cos_theta_(conf.getParameter<double>("min_cos_theta")),
      min_cos_pointing_(conf.getParameter<double>("min_cos_pointing")),
      etaLimitIncreaseWindow_(conf.getParameter<double>("etaLimitIncreaseWindow")),
      missing_layers_(conf.getParameter<int>("missing_layers")),
      min_clusters_per_ntuplet_(conf.getParameter<int>("min_clusters_per_ntuplet")),
      max_delta_time_(conf.getParameter<double>("max_delta_time")),
      eidInputName_(conf.getParameter<std::string>("eid_input_name")),
      eidOutputNameEnergy_(conf.getParameter<std::string>("eid_output_name_energy")),
      eidOutputNameId_(conf.getParameter<std::string>("eid_output_name_id")),
      eidMinClusterEnergy_(conf.getParameter<double>("eid_min_cluster_energy")),
      eidNLayers_(conf.getParameter<int>("eid_n_layers")),
      eidNClusters_(conf.getParameter<int>("eid_n_clusters")),
      eidSession_(nullptr) {
  // mount the tensorflow graph onto the session when set
  const TrackstersCache *trackstersCache = dynamic_cast<const TrackstersCache *>(cache);
  if (trackstersCache == nullptr || trackstersCache->eidGraphDef == nullptr) {
    throw cms::Exception("MissingGraphDef")
        << "PatternRecognitionbyCA received an empty graph definition from the global cache";
  }
  eidSession_ = tensorflow::createSession(trackstersCache->eidGraphDef);
}

template <typename TILES>
PatternRecognitionbyCA<TILES>::~PatternRecognitionbyCA(){};

template <typename TILES>
void PatternRecognitionbyCA<TILES>::makeTracksters(
    const typename PatternRecognitionAlgoBaseT<TILES>::Inputs &input,
    std::vector<Trackster> &result,
    std::unordered_map<int, std::vector<int>> &seedToTracksterAssociation) {
  // Protect from events with no seeding regions
  if (input.regions.empty())
    return;

  edm::ESHandle<CaloGeometry> geom;
  edm::EventSetup const &es = input.es;
  es.get<CaloGeometryRecord>().get(geom);
  rhtools_.setGeometry(*geom);

  theGraph_->setVerbosity(PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_);
  theGraph_->clear();
  if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > PatternRecognitionAlgoBaseT<TILES>::None) {
    LogDebug("HGCPatternRecoByCA") << "Making Tracksters with CA" << std::endl;
  }

  int type = input.tiles[0].typeT();
  int nEtaBin = (type == 1) ? ticl::TileConstantsHFNose::nEtaBins : ticl::TileConstants::nEtaBins;
  int nPhiBin = (type == 1) ? ticl::TileConstantsHFNose::nPhiBins : ticl::TileConstants::nPhiBins;

  bool isRegionalIter = (input.regions[0].index != -1);
  std::vector<HGCDoublet::HGCntuplet> foundNtuplets;
  std::vector<int> seedIndices;
  std::vector<uint8_t> layer_cluster_usage(input.layerClusters.size(), 0);
  theGraph_->makeAndConnectDoublets(input.tiles,
                                    input.regions,
                                    nEtaBin,
                                    nPhiBin,
                                    input.layerClusters,
                                    input.mask,
                                    input.layerClustersTime,
                                    1,
                                    1,
                                    min_cos_theta_,
                                    min_cos_pointing_,
                                    etaLimitIncreaseWindow_,
                                    missing_layers_,
                                    rhtools_.lastLayer(type),
                                    max_delta_time_);

  theGraph_->findNtuplets(foundNtuplets, seedIndices, min_clusters_per_ntuplet_, out_in_dfs_, max_out_in_hops_);
  //#ifdef FP_DEBUG
  const auto &doublets = theGraph_->getAllDoublets();
  int tracksterId = 0;

  for (auto const &ntuplet : foundNtuplets) {
    std::set<unsigned int> effective_cluster_idx;
    std::pair<std::set<unsigned int>::iterator, bool> retVal;

    std::vector<float> times;
    std::vector<float> timeErrors;

    for (auto const &doublet : ntuplet) {
      auto innerCluster = doublets[doublet].innerClusterId();
      auto outerCluster = doublets[doublet].outerClusterId();

      retVal = effective_cluster_idx.insert(innerCluster);
      if (retVal.second) {
        float time = input.layerClustersTime.get(innerCluster).first;
        if (time > -99) {
          times.push_back(time);
          timeErrors.push_back(1. / pow(input.layerClustersTime.get(innerCluster).second, 2));
        }
      }

      retVal = effective_cluster_idx.insert(outerCluster);
      if (retVal.second) {
        float time = input.layerClustersTime.get(outerCluster).first;
        if (time > -99) {
          times.push_back(time);
          timeErrors.push_back(1. / pow(input.layerClustersTime.get(outerCluster).second, 2));
        }
      }

      if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > PatternRecognitionAlgoBaseT<TILES>::Advanced) {
        LogDebug("HGCPatternRecoByCA") << " New doublet " << doublet << " for trackster: " << result.size()
                                       << " InnerCl " << innerCluster << " " << input.layerClusters[innerCluster].x()
                                       << " " << input.layerClusters[innerCluster].y() << " "
                                       << input.layerClusters[innerCluster].z() << " OuterCl " << outerCluster << " "
                                       << input.layerClusters[outerCluster].x() << " "
                                       << input.layerClusters[outerCluster].y() << " "
                                       << input.layerClusters[outerCluster].z() << " " << tracksterId << std::endl;
      }
    }
    for (auto const i : effective_cluster_idx) {
      layer_cluster_usage[i]++;
      if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > PatternRecognitionAlgoBaseT<TILES>::Basic)
        LogDebug("HGCPatternRecoByCA") << "LayerID: " << i << " count: " << (int)layer_cluster_usage[i] << std::endl;
    }
    // Put back indices, in the form of a Trackster, into the results vector
    Trackster tmp;
    tmp.vertices().reserve(effective_cluster_idx.size());
    tmp.vertex_multiplicity().resize(effective_cluster_idx.size(), 0);
    //regions and seedIndices can have different size
    //if a seeding region does not lead to any trackster
    tmp.setSeed(input.regions[0].collectionID, seedIndices[tracksterId]);
    if (isRegionalIter) {
      seedToTracksterAssociation[tmp.seedIndex()].push_back(tracksterId);
    }

    std::pair<float, float> timeTrackster(-99., -1.);
    hgcalsimclustertime::ComputeClusterTime timeEstimator;
    timeTrackster = timeEstimator.fixSizeHighestDensity(times, timeErrors);
    tmp.setTimeAndError(timeTrackster.first, timeTrackster.second);
    std::copy(std::begin(effective_cluster_idx), std::end(effective_cluster_idx), std::back_inserter(tmp.vertices()));
    result.push_back(tmp);
    tracksterId++;
  }

  for (auto &trackster : result) {
    assert(trackster.vertices().size() <= trackster.vertex_multiplicity().size());
    for (size_t i = 0; i < trackster.vertices().size(); ++i) {
      trackster.vertex_multiplicity()[i] = layer_cluster_usage[trackster.vertices(i)];
      if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > PatternRecognitionAlgoBaseT<TILES>::Basic)
        LogDebug("HGCPatternRecoByCA") << "LayerID: " << trackster.vertices(i)
                                       << " count: " << (int)trackster.vertex_multiplicity(i) << std::endl;
    }
  }

  // Now decide if the tracksters from the track-based iterations have to be merged
  if (oneTracksterPerTrackSeed_) {
    std::vector<Trackster> tmp;
    mergeTrackstersTRK(result, input.layerClusters, tmp, seedToTracksterAssociation);
    tmp.swap(result);
  }

  ticl::assignPCAtoTracksters(result, input.layerClusters, rhtools_.getPositionLayer(rhtools_.lastLayerEE(type)).z());

  // run energy regression and ID
  energyRegressionAndID(input.layerClusters, result);

  // now adding dummy tracksters from seeds not connected to any shower in the result collection
  // these are marked as charged hadrons with probability 1.
  if (promoteEmptyRegionToTrackster_) {
    emptyTrackstersFromSeedsTRK(result, seedToTracksterAssociation, input.regions[0].collectionID);
  }

  if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > PatternRecognitionAlgoBaseT<TILES>::Advanced) {
    for (auto &trackster : result) {
      LogDebug("HGCPatternRecoByCA") << "Trackster characteristics: " << std::endl;
      LogDebug("HGCPatternRecoByCA") << "Size: " << trackster.vertices().size() << std::endl;
      auto counter = 0;
      for (auto const &p : trackster.id_probabilities()) {
        LogDebug("HGCPatternRecoByCA") << counter++ << ": " << p << std::endl;
      }
    }
  }
  theGraph_->clear();
}

template <typename TILES>
void PatternRecognitionbyCA<TILES>::mergeTrackstersTRK(
    const std::vector<Trackster> &input,
    const std::vector<reco::CaloCluster> &layerClusters,
    std::vector<Trackster> &output,
    std::unordered_map<int, std::vector<int>> &seedToTracksterAssociation) const {
  output.reserve(input.size());
  for (auto &thisSeed : seedToTracksterAssociation) {
    auto &tracksters = thisSeed.second;
    if (!tracksters.empty()) {
      auto numberOfTrackstersInSeed = tracksters.size();
      output.emplace_back(input[tracksters[0]]);
      auto &outTrackster = output.back();
      tracksters[0] = output.size() - 1;
      auto updated_size = outTrackster.vertices().size();
      for (unsigned int j = 1; j < numberOfTrackstersInSeed; ++j) {
        auto &thisTrackster = input[tracksters[j]];
        updated_size += thisTrackster.vertices().size();
        if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > PatternRecognitionAlgoBaseT<TILES>::Basic) {
          LogDebug("HGCPatternRecoByCA") << "Updated size: " << updated_size << std::endl;
        }
        outTrackster.vertices().reserve(updated_size);
        outTrackster.vertex_multiplicity().reserve(updated_size);
        std::copy(std::begin(thisTrackster.vertices()),
                  std::end(thisTrackster.vertices()),
                  std::back_inserter(outTrackster.vertices()));
        std::copy(std::begin(thisTrackster.vertex_multiplicity()),
                  std::end(thisTrackster.vertex_multiplicity()),
                  std::back_inserter(outTrackster.vertex_multiplicity()));
      }
      tracksters.resize(1);
    }
  }
  output.shrink_to_fit();
}

template <typename TILES>
void PatternRecognitionbyCA<TILES>::emptyTrackstersFromSeedsTRK(
    std::vector<Trackster> &tracksters,
    std::unordered_map<int, std::vector<int>> &seedToTracksterAssociation,
    const edm::ProductID &collectionID) const {
  for (auto &thisSeed : seedToTracksterAssociation) {
    if (thisSeed.second.empty()) {
      Trackster t;
      t.setRegressedEnergy(0.f);
      t.zeroProbabilities();
      t.setIdProbability(ticl::Trackster::ParticleType::charged_hadron, 1.f);
      t.setSeed(collectionID, thisSeed.first);
      tracksters.emplace_back(t);
      thisSeed.second.emplace_back(tracksters.size() - 1);
    }
  }
}

template <typename TILES>
void PatternRecognitionbyCA<TILES>::energyRegressionAndID(const std::vector<reco::CaloCluster> &layerClusters,
                                                          std::vector<Trackster> &tracksters) {
  // Energy regression and particle identification strategy:
  //
  // 1. Set default values for regressed energy and particle id for each trackster.
  // 2. Store indices of tracksters whose total sum of cluster energies is above the
  //    eidMinClusterEnergy_ (GeV) treshold. Inference is not applied for soft tracksters.
  // 3. When no trackster passes the selection, return.
  // 4. Create input and output tensors. The batch dimension is determined by the number of
  //    selected tracksters.
  // 5. Fill input tensors with layer cluster features. Per layer, clusters are ordered descending
  //    by energy. Given that tensor data is contiguous in memory, we can use pointer arithmetic to
  //    fill values, even with batching.
  // 6. Zero-fill features for empty clusters in each layer.
  // 7. Batched inference.
  // 8. Assign the regressed energy and id probabilities to each trackster.
  //
  // Indices used throughout this method:
  // i -> batch element / trackster
  // j -> layer
  // k -> cluster
  // l -> feature

  // set default values per trackster, determine if the cluster energy threshold is passed,
  // and store indices of hard tracksters
  std::vector<int> tracksterIndices;
  for (int i = 0; i < (int)tracksters.size(); i++) {
    // calculate the cluster energy sum (2)
    // note: after the loop, sumClusterEnergy might be just above the threshold which is enough to
    // decide whether to run inference for the trackster or not
    float sumClusterEnergy = 0.;
    for (const unsigned int &vertex : tracksters[i].vertices()) {
      sumClusterEnergy += (float)layerClusters[vertex].energy();
      // there might be many clusters, so try to stop early
      if (sumClusterEnergy >= eidMinClusterEnergy_) {
        // set default values (1)
        tracksters[i].setRegressedEnergy(0.f);
        tracksters[i].zeroProbabilities();
        tracksterIndices.push_back(i);
        break;
      }
    }
  }

  // do nothing when no trackster passes the selection (3)
  int batchSize = (int)tracksterIndices.size();
  if (batchSize == 0) {
    return;
  }

  // create input and output tensors (4)
  tensorflow::TensorShape shape({batchSize, eidNLayers_, eidNClusters_, eidNFeatures_});
  tensorflow::Tensor input(tensorflow::DT_FLOAT, shape);
  tensorflow::NamedTensorList inputList = {{eidInputName_, input}};

  std::vector<tensorflow::Tensor> outputs;
  std::vector<std::string> outputNames;
  if (!eidOutputNameEnergy_.empty()) {
    outputNames.push_back(eidOutputNameEnergy_);
  }
  if (!eidOutputNameId_.empty()) {
    outputNames.push_back(eidOutputNameId_);
  }

  // fill input tensor (5)
  for (int i = 0; i < batchSize; i++) {
    const Trackster &trackster = tracksters[tracksterIndices[i]];

    // per layer, we only consider the first eidNClusters_ clusters in terms of energy, so in order
    // to avoid creating large / nested structures to do the sorting for an unknown number of total
    // clusters, create a sorted list of layer cluster indices to keep track of the filled clusters
    std::vector<int> clusterIndices(trackster.vertices().size());
    for (int k = 0; k < (int)trackster.vertices().size(); k++) {
      clusterIndices[k] = k;
    }
    sort(clusterIndices.begin(), clusterIndices.end(), [&layerClusters, &trackster](const int &a, const int &b) {
      return layerClusters[trackster.vertices(a)].energy() > layerClusters[trackster.vertices(b)].energy();
    });

    // keep track of the number of seen clusters per layer
    std::vector<int> seenClusters(eidNLayers_);

    // loop through clusters by descending energy
    for (const int &k : clusterIndices) {
      // get features per layer and cluster and store the values directly in the input tensor
      const reco::CaloCluster &cluster = layerClusters[trackster.vertices(k)];
      int j = rhtools_.getLayerWithOffset(cluster.hitsAndFractions()[0].first) - 1;
      if (j < eidNLayers_ && seenClusters[j] < eidNClusters_) {
        // get the pointer to the first feature value for the current batch, layer and cluster
        float *features = &input.tensor<float, 4>()(i, j, seenClusters[j], 0);

        // fill features
        *(features++) = float(cluster.energy() / float(trackster.vertex_multiplicity(k)));
        *(features++) = float(std::abs(cluster.eta()));
        *(features) = float(cluster.phi());

        // increment seen clusters
        seenClusters[j]++;
      }
    }

    // zero-fill features of empty clusters in each layer (6)
    for (int j = 0; j < eidNLayers_; j++) {
      for (int k = seenClusters[j]; k < eidNClusters_; k++) {
        float *features = &input.tensor<float, 4>()(i, j, k, 0);
        for (int l = 0; l < eidNFeatures_; l++) {
          *(features++) = 0.f;
        }
      }
    }
  }

  // run the inference (7)
  tensorflow::run(eidSession_, inputList, outputNames, &outputs);

  // store regressed energy per trackster (8)
  if (!eidOutputNameEnergy_.empty()) {
    // get the pointer to the energy tensor, dimension is batch x 1
    float *energy = outputs[0].flat<float>().data();

    for (const int &i : tracksterIndices) {
      tracksters[i].setRegressedEnergy(*(energy++));
    }
  }

  // store id probabilities per trackster (8)
  if (!eidOutputNameId_.empty()) {
    // get the pointer to the id probability tensor, dimension is batch x id_probabilities.size()
    int probsIdx = eidOutputNameEnergy_.empty() ? 0 : 1;
    float *probs = outputs[probsIdx].flat<float>().data();

    for (const int &i : tracksterIndices) {
      tracksters[i].setProbabilities(probs);
      probs += tracksters[i].id_probabilities().size();
    }
  }
}

template class ticl::PatternRecognitionbyCA<TICLLayerTiles>;
template class ticl::PatternRecognitionbyCA<TICLLayerTilesHFNose>;
