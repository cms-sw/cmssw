// Author: Marco Rovere - marco.rovere@cern.ch
// Date: 04/2021
#include <algorithm>
#include <set>
#include <vector>

#include "tbb/task_arena.h"
#include "tbb/tbb.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "PatternRecognitionbyCLUE3D.h"

#include "TrackstersPCA.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/Framework/interface/EventSetup.h"

using namespace ticl;

template <typename TILES>
PatternRecognitionbyCLUE3D<TILES>::PatternRecognitionbyCLUE3D(const edm::ParameterSet &conf, const CacheBase *cache)
    : PatternRecognitionAlgoBaseT<TILES>(conf, cache),
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
        << "PatternRecognitionbyCLUE3D received an empty graph definition from the global cache";
  }
  eidSession_ = tensorflow::createSession(trackstersCache->eidGraphDef);
}

template <typename TILES>
PatternRecognitionbyCLUE3D<TILES>::~PatternRecognitionbyCLUE3D(){};

template <typename TILES>
void PatternRecognitionbyCLUE3D<TILES>::dumpTiles(const TILES &tiles) const {
  int type = tiles[0].typeT();
  int nEtaBin = (type == 1) ? ticl::TileConstantsHFNose::nEtaBins : ticl::TileConstants::nEtaBins;
  int nPhiBin = (type == 1) ? ticl::TileConstantsHFNose::nPhiBins : ticl::TileConstants::nPhiBins;
  for (int layer = 0; layer < 104; layer++) {
    for (int ieta = 0; ieta < nEtaBin; ieta++) {
      auto offset = ieta * nPhiBin;
      for (int phi = 0; phi < nPhiBin; phi++) {
        int iphi = ((phi % nPhiBin + nPhiBin) % nPhiBin);
        if (!tiles[layer][offset + iphi].empty()) {
          LogDebug("PatternRecogntionbyCLUE3D") << "Layer: " << layer << " ieta: " << ieta << " phi: " << phi << " "
                                                << tiles[layer][offset + iphi].size();
        }
      }
    }
  }
}

template <typename TILES>
void PatternRecognitionbyCLUE3D<TILES>::dumpClusters(
    const std::vector<std::pair<int, int>> &layerIdx2layerandSoa) const {
  for (unsigned int layer = 0; layer < clusters_.size(); layer++) {
    LogDebug("PatternRecogntionbyCLUE3D") << "On Layer " << layer;
    auto const &thisLayer = clusters_[layer];
    LogDebug("PatternRecogntionbyCLUE3D") << "X: ";
    for (auto v : thisLayer.x)
      LogDebug("PatternRecogntionbyCLUE3D") << v << ",";
    LogDebug("PatternRecogntionbyCLUE3D") << std::endl;
    LogDebug("PatternRecogntionbyCLUE3D") << "Y: ";
    for (auto v : thisLayer.y)
      LogDebug("PatternRecogntionbyCLUE3D") << v << ",";
    LogDebug("PatternRecogntionbyCLUE3D") << std::endl;
    LogDebug("PatternRecogntionbyCLUE3D") << "Eta: ";
    for (auto v : thisLayer.eta)
      LogDebug("PatternRecogntionbyCLUE3D") << v << ",";
    LogDebug("PatternRecogntionbyCLUE3D") << std::endl;
    LogDebug("PatternRecogntionbyCLUE3D") << "Phi: ";
    for (auto v : thisLayer.phi)
      LogDebug("PatternRecogntionbyCLUE3D") << v << ",";
    LogDebug("PatternRecogntionbyCLUE3D") << std::endl;
    LogDebug("PatternRecogntionbyCLUE3D") << "Cells: ";
    for (auto v : thisLayer.cells)
      LogDebug("PatternRecogntionbyCLUE3D") << v << ",";
    LogDebug("PatternRecogntionbyCLUE3D") << std::endl;
    LogDebug("PatternRecogntionbyCLUE3D") << "Energy: ";
    for (auto v : thisLayer.energy)
      LogDebug("PatternRecogntionbyCLUE3D") << v << ",";
    LogDebug("PatternRecogntionbyCLUE3D") << std::endl;
    LogDebug("PatternRecogntionbyCLUE3D") << "Radius: ";
    for (auto v : thisLayer.radius)
      LogDebug("PatternRecogntionbyCLUE3D") << v << ",";
    LogDebug("PatternRecogntionbyCLUE3D") << std::endl;
    LogDebug("PatternRecogntionbyCLUE3D") << "Rho: ";
    for (auto v : thisLayer.rho)
      LogDebug("PatternRecogntionbyCLUE3D") << v << ",";
    LogDebug("PatternRecogntionbyCLUE3D") << std::endl;
    LogDebug("PatternRecogntionbyCLUE3D") << "Delta: ";
    for (auto v : thisLayer.delta)
      LogDebug("PatternRecogntionbyCLUE3D") << v << ",";
    LogDebug("PatternRecogntionbyCLUE3D") << std::endl;
    LogDebug("PatternRecogntionbyCLUE3D") << "ClusterIdx: ";
    for (auto v : thisLayer.clusterIndex)
      LogDebug("PatternRecogntionbyCLUE3D") << v << ",";
    LogDebug("PatternRecogntionbyCLUE3D") << std::endl;
    LogDebug("PatternRecogntionbyCLUE3D") << "LayerClusterOriginalIdx: ";
    for (auto v : thisLayer.layerClusterOriginalIdx)
      LogDebug("PatternRecogntionbyCLUE3D") << v << ",";
    LogDebug("PatternRecogntionbyCLUE3D") << std::endl;
  }
  for (unsigned int lcIdx = 0; lcIdx < layerIdx2layerandSoa.size(); lcIdx++) {
    auto const &layerandSoa = layerIdx2layerandSoa[lcIdx];
    LogDebug("PatternRecogntionbyCLUE3D")
        << "lcIdx: " << lcIdx << " on Layer: " << layerandSoa.first << " SOA: " << layerandSoa.second;
  }
}

template <typename TILES>
void PatternRecognitionbyCLUE3D<TILES>::makeTracksters(
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

  clusters_.clear();
  clusters_.resize(104);  // FIXME(rovere): Get it from template type or via rechittools
  std::vector<std::pair<int, int>> layerIdx2layerandSoa;

  layerIdx2layerandSoa.reserve(input.layerClusters.size());
  unsigned int layerIdx = 0;
  for (auto const &lc : input.layerClusters) {
    const auto firstHitDetId = lc.hitsAndFractions()[0].first;
    int layer = rhtools_.getLayerWithOffset(firstHitDetId) - 1 +
                rhtools_.lastLayer(false) * ((rhtools_.zside(firstHitDetId) + 1) >> 1);
    assert(layer >= 0);

    layerIdx2layerandSoa.emplace_back(layer, clusters_[layer].x.size());
    float sum_x = 0.;
    float sum_y = 0.;
    float sum_sqr_x = 0.;
    float sum_sqr_y = 0.;
    float ref_x = lc.x();
    float ref_y = lc.y();
    int clsize = lc.hitsAndFractions().size();
    for (auto const &hitsAndFractions : lc.hitsAndFractions()) {
      auto const &point = rhtools_.getPosition(hitsAndFractions.first);
      sum_x += point.x() - ref_x;
      sum_sqr_x += (point.x() - ref_x) * (point.x() - ref_x);
      sum_y += point.y() - ref_y;
      sum_sqr_y += (point.y() - ref_y) * (point.y() - ref_y);
    }
    // The variance of X for X uniform in circle of radius R is R^2/4,
    // therefore we multiply the sqrt(var) by 2 to have a rough estimate of the
    // radius. On the other hand, while averaging the x and y radius, we would
    // end up dividing by 2. Hence we omit the value here and in the average
    // below, too.
    float radius_x = sqrt((sum_sqr_x - (sum_x * sum_x) / clsize) / clsize);
    float radius_y = sqrt((sum_sqr_y - (sum_y * sum_y) / clsize) / clsize);
    clusters_[layer].x.emplace_back(lc.x());
    clusters_[layer].y.emplace_back(lc.y());
    clusters_[layer].radius.emplace_back(radius_x + radius_y);
    clusters_[layer].eta.emplace_back(lc.eta());
    clusters_[layer].phi.emplace_back(lc.phi());
    clusters_[layer].cells.push_back(lc.hitsAndFractions().size());
    clusters_[layer].energy.emplace_back(lc.energy());
    clusters_[layer].isSeed.push_back(-1);
    clusters_[layer].clusterIndex.emplace_back(-1);
    clusters_[layer].layerClusterOriginalIdx.emplace_back(layerIdx++);
    clusters_[layer].nearestHigher.emplace_back(-1, -1);
    clusters_[layer].rho.emplace_back(0.f);
    clusters_[layer].delta.emplace_back(std::numeric_limits<float>::max());
  }
  for (unsigned int layer = 0; layer < clusters_.size(); layer++) {
    clusters_[layer].followers.resize(clusters_[layer].x.size());
  }

  //  dumpTiles(input.tiles);
  std::vector<int> numberOfClustersPerLayer(104, 0);
  // tbb::this_task_arena::isolate([&] {
  //    tbb::parallel_for(size_t(0), size_t(104), [&](size_t i) { //FIXME(rovere): layer limits
  for (unsigned int i = 0; i < 104; i++) {
    calculateLocalDensity(input.tiles, i, layerIdx2layerandSoa);
    calculateDistanceToHigher(input.tiles, i, layerIdx2layerandSoa);
  }

  auto nTracksters = findAndAssignTracksters(input.tiles, layerIdx2layerandSoa);
  LogDebug("PatternRecogntionbyCLUE3D") << "Reconstructed " << nTracksters << " tracksters" << std::endl;
  dumpClusters(layerIdx2layerandSoa);
  //    );
  //  });

  // Build Trackster
  result.resize(nTracksters);

  assert(clusters_.size() == 104);
  for (unsigned int layer = 0; layer < clusters_.size(); ++layer) {
    const auto &thisLayer = clusters_[layer];
    LogDebug("PatternRecogntionbyCLUE3D") << "Examining Layer: " << layer;
    for (unsigned int lc = 0; lc < thisLayer.x.size(); ++lc) {
      LogDebug("PatternRecogntionbyCLUE3D") << "Trackster " << thisLayer.clusterIndex[lc];
      if (thisLayer.clusterIndex[lc] >= 0) {
        LogDebug("PatternRecogntionbyCLUE3D") << " adding lcIdx: " << thisLayer.layerClusterOriginalIdx[lc];
        result[thisLayer.clusterIndex[lc]].vertices().push_back(thisLayer.layerClusterOriginalIdx[lc]);
        result[thisLayer.clusterIndex[lc]].vertex_multiplicity().push_back(1);
      }
    }
  }

  //  // run energy regression and ID
  //  energyRegressionAndID(input.layerClusters, tmpTracksters);
  ticl::assignPCAtoTracksters(
      result, input.layerClusters, input.layerClustersTime, rhtools_.getPositionLayer(rhtools_.lastLayerEE(0), 0).z());

  // run energy regression and ID
  energyRegressionAndID(input.layerClusters, result);
  for (auto const &t : result) {
    LogDebug("PatternRecogntionbyCLUE3D") << "Barycenter: " << t.barycenter();
    LogDebug("PatternRecogntionbyCLUE3D") << "LCs: " << t.vertices().size();
    LogDebug("PatternRecogntionbyCLUE3D") << "Energy: " << t.raw_energy();
    LogDebug("PatternRecogntionbyCLUE3D") << "Regressed: " << t.regressed_energy();
  }

  // Reset internal clusters_ structure of array for next event
  reset();
  LogDebug("PatternRecogntionbyCLUE3D") << std::endl;
}

template <typename TILES>
void PatternRecognitionbyCLUE3D<TILES>::energyRegressionAndID(const std::vector<reco::CaloCluster> &layerClusters,
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

template <typename TILES>
void PatternRecognitionbyCLUE3D<TILES>::calculateLocalDensity(
    const TILES &tiles, const unsigned int layerId, const std::vector<std::pair<int, int>> &layerIdx2layerandSoa) {
  int type = tiles[0].typeT();
  int nEtaBin = (type == 1) ? ticl::TileConstantsHFNose::nEtaBins : ticl::TileConstants::nEtaBins;
  int nPhiBin = (type == 1) ? ticl::TileConstantsHFNose::nPhiBins : ticl::TileConstants::nPhiBins;
  auto &clustersOnLayer = clusters_[layerId];
  unsigned int numberOfClusters = clustersOnLayer.x.size();

  auto isReachable = [](float x1, float x2, float y1, float y2, float delta_sqr) -> bool {
    LogDebug("PatternRecogntionbyCLUE3D")
        << ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)) << " vs " << delta_sqr << "["
        << ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) < delta_sqr) << "]\n";
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) < delta_sqr;
  };

  for (unsigned int i = 0; i < numberOfClusters; i++) {
    unsigned int minLayer = std::max((int)layerId - 1, 0);
    unsigned int maxLayer = std::min((int)layerId + 1, 103);
    for (unsigned int currentLayer = minLayer; currentLayer <= maxLayer; currentLayer++) {
      LogDebug("PatternRecogntionbyCLUE3D") << "RefLayer: " << layerId << " SoaIDX: " << i;
      LogDebug("PatternRecogntionbyCLUE3D") << "NextLayer: " << currentLayer;
      const auto &tileOnLayer = tiles[currentLayer];
      bool onSameLayer = (currentLayer == layerId);
      LogDebug("PatternRecogntionbyCLUE3D") << "onSameLayer: " << onSameLayer;
      int etaWindow = onSameLayer ? 2 : 1;
      int phiWindow = onSameLayer ? 2 : 1;
      int etaBinMin = std::max(tileOnLayer.etaBin(clustersOnLayer.eta[i]) - etaWindow, 0);
      int etaBinMax = std::min(tileOnLayer.etaBin(clustersOnLayer.eta[i]) + etaWindow, nEtaBin);
      int phiBinMin = tileOnLayer.phiBin(clustersOnLayer.phi[i]) - phiWindow;
      int phiBinMax = tileOnLayer.phiBin(clustersOnLayer.phi[i]) + phiWindow;
      LogDebug("PatternRecogntionbyCLUE3D") << "eta: " << clustersOnLayer.eta[i];
      LogDebug("PatternRecogntionbyCLUE3D") << "phi: " << clustersOnLayer.phi[i];
      LogDebug("PatternRecogntionbyCLUE3D") << "etaBinMin: " << etaBinMin << ", etaBinMax: " << etaBinMax;
      LogDebug("PatternRecogntionbyCLUE3D") << "phiBinMin: " << phiBinMin << ", phiBinMax: " << phiBinMax;
      for (int ieta = etaBinMin; ieta < etaBinMax; ++ieta) {
        auto offset = ieta * nPhiBin;
        LogDebug("PatternRecogntionbyCLUE3D") << "offset: " << offset;
        for (int iphi_it = phiBinMin; iphi_it < phiBinMax; ++iphi_it) {
          int iphi = ((iphi_it % nPhiBin + nPhiBin) % nPhiBin);
          LogDebug("PatternRecogntionbyCLUE3D") << "iphi: " << iphi;
          LogDebug("PatternRecogntionbyCLUE3D") << "Entries in tileBin: " << tileOnLayer[offset + iphi].size();
          for (auto otherClusterIdx : tileOnLayer[offset + iphi]) {
            auto const &layerandSoa = layerIdx2layerandSoa[otherClusterIdx];
            LogDebug("PatternRecogntionbyCLUE3D")
                << "OtherLayer: " << layerandSoa.first << " SoaIDX: " << layerandSoa.second;
            LogDebug("PatternRecogntionbyCLUE3D")
                << "OtherEta: " << clusters_[layerandSoa.first].eta[layerandSoa.second];
            LogDebug("PatternRecogntionbyCLUE3D")
                << "OtherPhi: " << clusters_[layerandSoa.first].phi[layerandSoa.second];
            float delta = clustersOnLayer.radius[i] + clusters_[layerandSoa.first].radius[layerandSoa.second] +
                          2.6;  // 26 mm, roughly 2 cells, more wrt sum of radii
            if (onSameLayer) {
              if (isReachable(clustersOnLayer.x[i],
                              clusters_[layerandSoa.first].x[layerandSoa.second],
                              clustersOnLayer.y[i],
                              clusters_[layerandSoa.first].y[layerandSoa.second],
                              delta * delta)) {
                clustersOnLayer.rho[i] += (clustersOnLayer.layerClusterOriginalIdx[i] == otherClusterIdx ? 1.f : 0.5f) *
                                          clusters_[layerandSoa.first].energy[layerandSoa.second];
              }
            } else {
              if (isReachable(clustersOnLayer.eta[i],
                              clusters_[layerandSoa.first].eta[layerandSoa.second],
                              clustersOnLayer.phi[i],
                              clusters_[layerandSoa.first].phi[layerandSoa.second],
                              0.0025)) {
                clustersOnLayer.rho[i] += (clustersOnLayer.layerClusterOriginalIdx[i] == otherClusterIdx ? 1.f : 0.5f) *
                                          clusters_[layerandSoa.first].energy[layerandSoa.second];
              }
            }
          }
        }
      }
    }
  }
  LogDebug("PatternRecogntionbyCLUE3D") << std::endl;
}

template <typename TILES>
void PatternRecognitionbyCLUE3D<TILES>::calculateDistanceToHigher(
    const TILES &tiles, const unsigned int layerId, const std::vector<std::pair<int, int>> &layerIdx2layerandSoa) {
  int type = tiles[0].typeT();
  int nEtaBin = (type == 1) ? ticl::TileConstantsHFNose::nEtaBins : ticl::TileConstants::nEtaBins;
  int nPhiBin = (type == 1) ? ticl::TileConstantsHFNose::nPhiBins : ticl::TileConstants::nPhiBins;
  auto &clustersOnLayer = clusters_[layerId];
  unsigned int numberOfClusters = clustersOnLayer.x.size();

  auto distance = [](float x1, float x2, float y1, float y2) -> float {
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
  };

  for (unsigned int i = 0; i < numberOfClusters; i++) {
    unsigned int minLayer = std::max((int)layerId - 1, 0);
    unsigned int maxLayer = std::min((int)layerId + 1, 103);
    //FIXME(rovere): implement wrapping between +Z and -Z to avoid mixing last
    //of one with first of the other
    float maxDelta = std::numeric_limits<float>::max();
    float i_delta = maxDelta;
    std::pair<int, int> i_nearestHigher(-1, -1);
    for (unsigned int currentLayer = minLayer; currentLayer <= maxLayer; currentLayer++) {
      const auto &tileOnLayer = tiles[currentLayer];
      int etaWindow = 2;  //onSameLayer ? 2 : 1;
      int phiWindow = 2;  //onSameLayer ? 2 : 1;
      int etaBinMin = std::max(tileOnLayer.etaBin(clustersOnLayer.eta[i]) - etaWindow, 0);
      int etaBinMax = std::min(tileOnLayer.etaBin(clustersOnLayer.eta[i]) + etaWindow, nEtaBin);
      int phiBinMin = tileOnLayer.phiBin(clustersOnLayer.phi[i]) - phiWindow;
      int phiBinMax = tileOnLayer.phiBin(clustersOnLayer.phi[i]) + phiWindow;
      for (int ieta = etaBinMin; ieta < etaBinMax; ++ieta) {
        auto offset = ieta * nPhiBin;
        for (int iphi_it = phiBinMin; iphi_it < phiBinMax; ++iphi_it) {
          int iphi = ((iphi_it % nPhiBin + nPhiBin) % nPhiBin);
          for (auto otherClusterIdx : tileOnLayer[offset + iphi]) {
            auto const &layerandSoa = layerIdx2layerandSoa[otherClusterIdx];
            auto &clustersOnOtherLayer = clusters_[layerandSoa.first];
            float dist = distance(clustersOnLayer.eta[i],
                                  clustersOnOtherLayer.eta[layerandSoa.second],
                                  clustersOnLayer.phi[i],
                                  clustersOnOtherLayer.phi[layerandSoa.second]);
            bool foundHigher = (clustersOnOtherLayer.rho[layerandSoa.second] > clustersOnLayer.rho[i]) ||
                               (clustersOnOtherLayer.rho[layerandSoa.second] == clustersOnLayer.rho[i] &&
                                clustersOnOtherLayer.layerClusterOriginalIdx[layerandSoa.second] >
                                    clustersOnLayer.layerClusterOriginalIdx[i]);
            if (foundHigher && dist <= i_delta) {
              // update i_delta
              i_delta = dist;
              // update i_nearestHigher
              i_nearestHigher = (layerandSoa);
            }
          }
        }
      }
    }

    bool foundNearestHigherInEtaPhiCylinder = (i_delta <= 0.05);
    LogDebug("PatternRecogntionbyCLUE3D") << "i_delta: " << i_delta << " passed: " << (i_delta <= 0.05);
    if (foundNearestHigherInEtaPhiCylinder) {
      clustersOnLayer.delta[i] = i_delta;
      clustersOnLayer.nearestHigher[i] = i_nearestHigher;
    } else {
      // otherwise delta is guaranteed to be larger outlierDeltaFactor_*delta_c
      // we can safely maximize delta to be maxDelta
      clustersOnLayer.delta[i] = maxDelta;
      clustersOnLayer.nearestHigher[i] = {-1, -1};
    }
  }
}

template <typename TILES>
int PatternRecognitionbyCLUE3D<TILES>::findAndAssignTracksters(
    const TILES &tiles, const std::vector<std::pair<int, int>> &layerIdx2layerandSoa) {
  unsigned int nTracksters = 0;

  std::vector<std::pair<int, int>> localStack;
  // find cluster seeds and outlier
  // FIXME(rovere): loop can be parallel for with local stack vector addressed by layer
  for (unsigned int layer = 0; layer < 104; layer++) {
    auto &clustersOnLayer = clusters_[layer];
    unsigned int numberOfClusters = clustersOnLayer.x.size();
    for (unsigned int i = 0; i < numberOfClusters; i++) {
      float rho_c = 8.25f;
      float delta = 0.1;
      float outlierDeltaFactor = 1.1;

      // initialize clusterIndex
      clustersOnLayer.clusterIndex[i] = -1;
      bool isSeed = (clustersOnLayer.delta[i] > delta) && (clustersOnLayer.rho[i] >= rho_c);
      bool isOutlier = (clustersOnLayer.delta[i] > outlierDeltaFactor * delta) && (clustersOnLayer.rho[i] < rho_c);
      if (isSeed) {
        clustersOnLayer.clusterIndex[i] = nTracksters++;
        clustersOnLayer.isSeed[i] = true;
        localStack.emplace_back(layer, i);

      } else if (!isOutlier) {
        auto [lyrIdx, soaIdx] = clustersOnLayer.nearestHigher[i];
        clusters_[lyrIdx].followers[soaIdx].emplace_back(layer, i);
        //        clustersOnLayer.followers[clustersOnLayer.nearestHigher[i]].push_back(i);
      }
    }
  }

  // Propagate cluster index
  while (!localStack.empty()) {
    auto [lyrIdx, soaIdx] = localStack.back();
    auto &thisSeed = clusters_[lyrIdx].followers[soaIdx];
    localStack.pop_back();

    // loop over followers
    for (auto [follower_lyrIdx, follower_soaIdx] : thisSeed) {
      // pass id to a follower
      clusters_[follower_lyrIdx].clusterIndex[follower_soaIdx] = clusters_[lyrIdx].clusterIndex[soaIdx];
      // push this follower to localStack
      localStack.emplace_back(follower_lyrIdx, follower_soaIdx);
    }
  }
  return nTracksters;
}

template class ticl::PatternRecognitionbyCLUE3D<TICLLayerTiles>;
template class ticl::PatternRecognitionbyCLUE3D<TICLLayerTilesHFNose>;
