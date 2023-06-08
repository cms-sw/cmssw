// Author: Marco Rovere - marco.rovere@cern.ch
// Date: 04/2021
#include <algorithm>
#include <set>
#include <vector>

#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "PatternRecognitionbyCLUE3D.h"

#include "TrackstersPCA.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/Framework/interface/EventSetup.h"

using namespace ticl;

template <typename TILES>
PatternRecognitionbyCLUE3D<TILES>::PatternRecognitionbyCLUE3D(const edm::ParameterSet &conf, edm::ConsumesCollector iC)
    : PatternRecognitionAlgoBaseT<TILES>(conf, iC),
      caloGeomToken_(iC.esConsumes<CaloGeometry, CaloGeometryRecord>()),
      criticalDensity_(conf.getParameter<double>("criticalDensity")),
      criticalSelfDensity_(conf.getParameter<double>("criticalSelfDensity")),
      densitySiblingLayers_(conf.getParameter<int>("densitySiblingLayers")),
      densityEtaPhiDistanceSqr_(conf.getParameter<double>("densityEtaPhiDistanceSqr")),
      densityXYDistanceSqr_(conf.getParameter<double>("densityXYDistanceSqr")),
      kernelDensityFactor_(conf.getParameter<double>("kernelDensityFactor")),
      densityOnSameLayer_(conf.getParameter<bool>("densityOnSameLayer")),
      nearestHigherOnSameLayer_(conf.getParameter<bool>("nearestHigherOnSameLayer")),
      useAbsoluteProjectiveScale_(conf.getParameter<bool>("useAbsoluteProjectiveScale")),
      useClusterDimensionXY_(conf.getParameter<bool>("useClusterDimensionXY")),
      rescaleDensityByZ_(conf.getParameter<bool>("rescaleDensityByZ")),
      criticalEtaPhiDistance_(conf.getParameter<double>("criticalEtaPhiDistance")),
      criticalXYDistance_(conf.getParameter<double>("criticalXYDistance")),
      criticalZDistanceLyr_(conf.getParameter<int>("criticalZDistanceLyr")),
      outlierMultiplier_(conf.getParameter<double>("outlierMultiplier")),
      minNumLayerCluster_(conf.getParameter<int>("minNumLayerCluster")),
      eidInputName_(conf.getParameter<std::string>("eid_input_name")),
      eidOutputNameEnergy_(conf.getParameter<std::string>("eid_output_name_energy")),
      eidOutputNameId_(conf.getParameter<std::string>("eid_output_name_id")),
      eidMinClusterEnergy_(conf.getParameter<double>("eid_min_cluster_energy")),
      eidNLayers_(conf.getParameter<int>("eid_n_layers")),
      eidNClusters_(conf.getParameter<int>("eid_n_clusters")){};

template <typename TILES>
void PatternRecognitionbyCLUE3D<TILES>::dumpTiles(const TILES &tiles) const {
  constexpr int nEtaBin = TILES::constants_type_t::nEtaBins;
  constexpr int nPhiBin = TILES::constants_type_t::nPhiBins;
  auto lastLayerPerSide = static_cast<int>(rhtools_.lastLayer(false));
  int maxLayer = 2 * lastLayerPerSide - 1;
  for (int layer = 0; layer <= maxLayer; layer++) {
    for (int ieta = 0; ieta < nEtaBin; ieta++) {
      auto offset = ieta * nPhiBin;
      for (int phi = 0; phi < nPhiBin; phi++) {
        int iphi = ((phi % nPhiBin + nPhiBin) % nPhiBin);
        if (!tiles[layer][offset + iphi].empty()) {
          if (this->algo_verbosity_ > VerbosityLevel::Advanced) {
            edm::LogVerbatim("PatternRecognitionbyCLUE3D") << "Layer: " << layer << " ieta: " << ieta << " phi: " << phi
                                                           << " " << tiles[layer][offset + iphi].size();
          }
        }
      }
    }
  }
}

template <typename TILES>
void PatternRecognitionbyCLUE3D<TILES>::dumpTracksters(const std::vector<std::pair<int, int>> &layerIdx2layerandSoa,
                                                       const int eventNumber,
                                                       const std::vector<Trackster> &tracksters) const {
  if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
    edm::LogVerbatim("PatternRecognitionbyCLUE3D")
        << "[evt, tracksterId, cells, prob_photon, prob_ele, prob_chad, prob_nhad, layer_i, x_i, y_i, eta_i, phi_i, "
           "energy_i, radius_i, rho_i, z_extension, delta_tr, delta_lyr, isSeed_i";
  }

  int num = 0;
  const std::string sep(", ");
  for (auto const &t : tracksters) {
    for (auto v : t.vertices()) {
      auto [lyrIdx, soaIdx] = layerIdx2layerandSoa[v];
      auto const &thisLayer = clusters_[lyrIdx];
      if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
        edm::LogVerbatim("PatternRecognitionbyCLUE3D_NTP")
            << std::setw(4) << eventNumber << sep << std::setw(4) << num << sep << std::setw(4) << t.vertices().size()
            << sep << std::setw(8) << t.id_probability(ticl::Trackster::ParticleType::photon) << sep << std::setw(8)
            << t.id_probability(ticl::Trackster::ParticleType::electron) << sep << std::setw(8)
            << t.id_probability(ticl::Trackster::ParticleType::charged_hadron) << sep << std::setw(8)
            << t.id_probability(ticl::Trackster::ParticleType::neutral_hadron) << sep << std::setw(4) << lyrIdx << sep
            << std::setw(10) << thisLayer.x[soaIdx] << sep << std::setw(10) << thisLayer.y[soaIdx] << sep
            << std::setw(10) << thisLayer.eta[soaIdx] << sep << std::setw(10) << thisLayer.phi[soaIdx] << sep
            << std::setw(10) << thisLayer.energy[soaIdx] << sep << std::setw(10) << thisLayer.radius[soaIdx] << sep
            << std::setw(10) << thisLayer.rho[soaIdx] << sep << std::setw(10) << thisLayer.z_extension[soaIdx] << sep
            << std::setw(10) << thisLayer.delta[soaIdx].first << sep << std::setw(10) << thisLayer.delta[soaIdx].second
            << sep << std::setw(4) << thisLayer.isSeed[soaIdx];
      }
    }
    num++;
  }
}

template <typename TILES>
void PatternRecognitionbyCLUE3D<TILES>::dumpClusters(const TILES &tiles,
                                                     const std::vector<std::pair<int, int>> &layerIdx2layerandSoa,
                                                     const int eventNumber) const {
  if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
    edm::LogVerbatim("PatternRecognitionbyCLUE3D") << "[evt, lyr, Seed,      x,       y,       z, r/|z|,   eta,   phi, "
                                                      "etab,  phib, cells, enrgy, e/rho,   rho,   z_ext, "
                                                      "   dlt_tr,   dlt_lyr, "
                                                      " nestHL, nestHSoaIdx, radius, clIdx, lClOrigIdx, SOAidx";
  }

  for (unsigned int layer = 0; layer < clusters_.size(); layer++) {
    auto const &thisLayer = clusters_[layer];
    int num = 0;
    for (auto v : thisLayer.x) {
      if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
        edm::LogVerbatim("PatternRecognitionbyCLUE3D")
            << std::setw(4) << eventNumber << ", " << std::setw(3) << layer << ", " << std::setw(4)
            << thisLayer.isSeed[num] << ", " << std::setprecision(3) << std::fixed << v << ", " << thisLayer.y[num]
            << ", " << thisLayer.z[num] << ", " << thisLayer.r_over_absz[num] << ", " << thisLayer.eta[num] << ", "
            << thisLayer.phi[num] << ", " << std::setw(5) << tiles[layer].etaBin(thisLayer.eta[num]) << ", "
            << std::setw(5) << tiles[layer].phiBin(thisLayer.phi[num]) << ", " << std::setw(4) << thisLayer.cells[num]
            << ", " << std::setprecision(3) << thisLayer.energy[num] << ", "
            << (thisLayer.energy[num] / thisLayer.rho[num]) << ", " << thisLayer.rho[num] << ", "
            << thisLayer.z_extension[num] << ", " << std::scientific << thisLayer.delta[num].first << ", "
            << std::setw(10) << thisLayer.delta[num].second << ", " << std::setw(5)
            << thisLayer.nearestHigher[num].first << ", " << std::setw(10) << thisLayer.nearestHigher[num].second
            << ", " << std::defaultfloat << std::setprecision(3) << thisLayer.radius[num] << ", " << std::setw(5)
            << thisLayer.clusterIndex[num] << ", " << std::setw(4) << thisLayer.layerClusterOriginalIdx[num] << ", "
            << std::setw(4) << num << ", ClusterInfo";
      }
      ++num;
    }
  }
  for (unsigned int lcIdx = 0; lcIdx < layerIdx2layerandSoa.size(); lcIdx++) {
    auto const &layerandSoa = layerIdx2layerandSoa[lcIdx];
    // Skip masked layer clusters
    if ((layerandSoa.first == -1) && (layerandSoa.second == -1))
      continue;
    if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
      edm::LogVerbatim("PatternRecognitionbyCLUE3D")
          << "lcIdx: " << lcIdx << " on Layer: " << layerandSoa.first << " SOA: " << layerandSoa.second;
    }
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

  const int eventNumber = input.ev.eventAuxiliary().event();
  if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
    edm::LogVerbatim("PatternRecognitionbyCLUE3D") << "New Event";
  }

  edm::EventSetup const &es = input.es;
  const CaloGeometry &geom = es.getData(caloGeomToken_);
  rhtools_.setGeometry(geom);

  // Assume identical Z-positioning between positive and negative sides.
  // Also, layers inside the HGCAL geometry start from 1.
  for (unsigned int i = 0; i < rhtools_.lastLayer(); ++i) {
    layersPosZ_.push_back(rhtools_.getPositionLayer(i + 1).z());
    if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
      edm::LogVerbatim("PatternRecognitionbyCLUE3D") << "Layer " << i << " located at Z: " << layersPosZ_.back();
    }
  }

  clusters_.clear();
  clusters_.resize(2 * rhtools_.lastLayer(false));
  std::vector<std::pair<int, int>> layerIdx2layerandSoa;  //used everywhere also to propagate cluster masking

  layerIdx2layerandSoa.reserve(input.layerClusters.size());
  unsigned int layerIdx = 0;
  for (auto const &lc : input.layerClusters) {
    if (input.mask[layerIdx] == 0.) {
      if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
        edm::LogVerbatim("PatternRecognitionbyCLUE3D") << "Skipping masked cluster: " << layerIdx;
      }
      layerIdx2layerandSoa.emplace_back(-1, -1);
      layerIdx++;
      continue;
    }
    const auto firstHitDetId = lc.hitsAndFractions()[0].first;
    int layer = rhtools_.getLayerWithOffset(firstHitDetId) - 1 +
                rhtools_.lastLayer(false) * ((rhtools_.zside(firstHitDetId) + 1) >> 1);
    assert(layer >= 0);
    auto detId = lc.hitsAndFractions()[0].first;

    layerIdx2layerandSoa.emplace_back(layer, clusters_[layer].x.size());
    float sum_x = 0.;
    float sum_y = 0.;
    float sum_sqr_x = 0.;
    float sum_sqr_y = 0.;
    float ref_x = lc.x();
    float ref_y = lc.y();
    float invClsize = 1. / lc.hitsAndFractions().size();
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
    float radius_x = sqrt((sum_sqr_x - (sum_x * sum_x) * invClsize) * invClsize);
    float radius_y = sqrt((sum_sqr_y - (sum_y * sum_y) * invClsize) * invClsize);
    if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
      edm::LogVerbatim("PatternRecognitionbyCLUE3D")
          << "cluster rx: " << std::setw(5) << radius_x << ", ry: " << std::setw(5) << radius_y
          << ", r:  " << std::setw(5) << (radius_x + radius_y) << ", cells: " << std::setw(4)
          << lc.hitsAndFractions().size();
    }

    // The case of single cell layer clusters has to be handled differently.

    if (invClsize == 1.) {
      // Silicon case
      if (rhtools_.isSilicon(detId)) {
        radius_x = radius_y = rhtools_.getRadiusToSide(detId);
        if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
          edm::LogVerbatim("PatternRecognitionbyCLUE3D") << "Single cell cluster in silicon, rx: " << std::setw(5)
                                                         << radius_x << ", ry: " << std::setw(5) << radius_y;
        }
      } else {
        auto const &point = rhtools_.getPosition(detId);
        auto const &eta_phi_window = rhtools_.getScintDEtaDPhi(detId);
        radius_x = radius_y = point.perp() * eta_phi_window.second;
        if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
          edm::LogVerbatim("PatternRecognitionbyCLUE3D")
              << "Single cell cluster in scintillator. rx: " << std::setw(5) << radius_x << ", ry: " << std::setw(5)
              << radius_y << ", eta-span: " << std::setw(5) << eta_phi_window.first << ", phi-span: " << std::setw(5)
              << eta_phi_window.second;
        }
      }
    }
    clusters_[layer].x.emplace_back(lc.x());
    clusters_[layer].y.emplace_back(lc.y());
    clusters_[layer].z.emplace_back(lc.z());
    clusters_[layer].r_over_absz.emplace_back(sqrt(lc.x() * lc.x() + lc.y() * lc.y()) / std::abs(lc.z()));
    clusters_[layer].radius.emplace_back(radius_x + radius_y);
    clusters_[layer].eta.emplace_back(lc.eta());
    clusters_[layer].phi.emplace_back(lc.phi());
    clusters_[layer].cells.push_back(lc.hitsAndFractions().size());
    clusters_[layer].isSilicon.push_back(rhtools_.isSilicon(detId));
    clusters_[layer].energy.emplace_back(lc.energy());
    clusters_[layer].isSeed.push_back(false);
    clusters_[layer].clusterIndex.emplace_back(-1);
    clusters_[layer].layerClusterOriginalIdx.emplace_back(layerIdx++);
    clusters_[layer].nearestHigher.emplace_back(-1, -1);
    clusters_[layer].rho.emplace_back(0.f);
    clusters_[layer].z_extension.emplace_back(0.f);
    clusters_[layer].delta.emplace_back(
        std::make_pair(std::numeric_limits<float>::max(), std::numeric_limits<int>::max()));
  }
  for (unsigned int layer = 0; layer < clusters_.size(); layer++) {
    clusters_[layer].followers.resize(clusters_[layer].x.size());
  }

  auto lastLayerPerSide = static_cast<int>(rhtools_.lastLayer(false));
  int maxLayer = 2 * lastLayerPerSide - 1;
  std::vector<int> numberOfClustersPerLayer(maxLayer, 0);
  for (int i = 0; i <= maxLayer; i++) {
    calculateLocalDensity(input.tiles, i, layerIdx2layerandSoa);
  }
  for (int i = 0; i <= maxLayer; i++) {
    calculateDistanceToHigher(input.tiles, i, layerIdx2layerandSoa);
  }

  auto nTracksters = findAndAssignTracksters(input.tiles, layerIdx2layerandSoa);
  if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
    edm::LogVerbatim("PatternRecognitionbyCLUE3D") << "Reconstructed " << nTracksters << " tracksters" << std::endl;
    dumpClusters(input.tiles, layerIdx2layerandSoa, eventNumber);
  }

  // Build Trackster
  result.resize(nTracksters);

  for (unsigned int layer = 0; layer < clusters_.size(); ++layer) {
    const auto &thisLayer = clusters_[layer];
    if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
      edm::LogVerbatim("PatternRecognitionbyCLUE3D") << "Examining Layer: " << layer;
    }
    for (unsigned int lc = 0; lc < thisLayer.x.size(); ++lc) {
      if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
        edm::LogVerbatim("PatternRecognitionbyCLUE3D") << "Trackster " << thisLayer.clusterIndex[lc];
      }
      if (thisLayer.clusterIndex[lc] >= 0) {
        if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
          edm::LogVerbatim("PatternRecognitionbyCLUE3D") << " adding lcIdx: " << thisLayer.layerClusterOriginalIdx[lc];
        }
        result[thisLayer.clusterIndex[lc]].vertices().push_back(thisLayer.layerClusterOriginalIdx[lc]);
        result[thisLayer.clusterIndex[lc]].vertex_multiplicity().push_back(1);
        // loop over followers
        for (auto [follower_lyrIdx, follower_soaIdx] : thisLayer.followers[lc]) {
          std::array<unsigned int, 2> edge = {
              {(unsigned int)thisLayer.layerClusterOriginalIdx[lc],
               (unsigned int)clusters_[follower_lyrIdx].layerClusterOriginalIdx[follower_soaIdx]}};
          result[thisLayer.clusterIndex[lc]].edges().push_back(edge);
        }
      }
    }
  }

  result.erase(
      std::remove_if(std::begin(result),
                     std::end(result),
                     [&](auto const &v) { return static_cast<int>(v.vertices().size()) < minNumLayerCluster_; }),
      result.end());
  result.shrink_to_fit();

  ticl::assignPCAtoTracksters(result,
                              input.layerClusters,
                              input.layerClustersTime,
                              rhtools_.getPositionLayer(rhtools_.lastLayerEE(false), false).z());

  // run energy regression and ID
  energyRegressionAndID(input.layerClusters, input.tfSession, result);
  if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
    for (auto const &t : result) {
      edm::LogVerbatim("PatternRecognitionbyCLUE3D") << "Barycenter: " << t.barycenter();
      edm::LogVerbatim("PatternRecognitionbyCLUE3D") << "LCs: " << t.vertices().size();
      edm::LogVerbatim("PatternRecognitionbyCLUE3D") << "Energy: " << t.raw_energy();
      edm::LogVerbatim("PatternRecognitionbyCLUE3D") << "Regressed: " << t.regressed_energy();
    }
  }

  // Dump Tracksters information
  if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
    dumpTracksters(layerIdx2layerandSoa, eventNumber, result);
  }

  // Reset internal clusters_ structure of array for next event
  reset();
  if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
    edm::LogVerbatim("PatternRecognitionbyCLUE3D") << std::endl;
  }
}

template <typename TILES>
void PatternRecognitionbyCLUE3D<TILES>::energyRegressionAndID(const std::vector<reco::CaloCluster> &layerClusters,
                                                              const tensorflow::Session *eidSession,
                                                              std::vector<Trackster> &tracksters) {
  // Energy regression and particle identification strategy:
  //
  // 1. Set default values for regressed energy and particle id for each trackster.
  // 2. Store indices of tracksters whose total sum of cluster energies is above the
  //    eidMinClusterEnergy_ (GeV) threshold. Inference is not applied for soft tracksters.
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
  for (int i = 0; i < static_cast<int>(tracksters.size()); i++) {
    // calculate the cluster energy sum (2)
    // note: after the loop, sumClusterEnergy might be just above the threshold which is enough to
    // decide whether to run inference for the trackster or not
    float sumClusterEnergy = 0.;
    for (const unsigned int &vertex : tracksters[i].vertices()) {
      sumClusterEnergy += static_cast<float>(layerClusters[vertex].energy());
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
  int batchSize = static_cast<int>(tracksterIndices.size());
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
  tensorflow::run(eidSession, inputList, outputNames, &outputs);

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
    const TILES &tiles, const int layerId, const std::vector<std::pair<int, int>> &layerIdx2layerandSoa) {
  constexpr int nEtaBin = TILES::constants_type_t::nEtaBins;
  constexpr int nPhiBin = TILES::constants_type_t::nPhiBins;
  auto &clustersOnLayer = clusters_[layerId];
  unsigned int numberOfClusters = clustersOnLayer.x.size();

  auto isReachable = [](float r0, float r1, float phi0, float phi1, float delta_sqr) -> bool {
    auto delta_phi = reco::deltaPhi(phi0, phi1);
    return (r0 - r1) * (r0 - r1) + r1 * r1 * delta_phi * delta_phi < delta_sqr;
  };
  auto distance_debug = [&](float x1, float x2, float y1, float y2) -> float {
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
  };

  for (unsigned int i = 0; i < numberOfClusters; i++) {
    // We need to partition the two sides of the HGCAL detector
    auto lastLayerPerSide = static_cast<int>(rhtools_.lastLayer(false));
    int minLayer = 0;
    int maxLayer = 2 * lastLayerPerSide - 1;
    if (layerId < lastLayerPerSide) {
      minLayer = std::max(layerId - densitySiblingLayers_, minLayer);
      maxLayer = std::min(layerId + densitySiblingLayers_, lastLayerPerSide - 1);
    } else {
      minLayer = std::max(layerId - densitySiblingLayers_, lastLayerPerSide);
      maxLayer = std::min(layerId + densitySiblingLayers_, maxLayer);
    }
    float deltaLayersZ = std::abs(layersPosZ_[maxLayer % lastLayerPerSide] - layersPosZ_[minLayer % lastLayerPerSide]);

    for (int currentLayer = minLayer; currentLayer <= maxLayer; currentLayer++) {
      if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
        edm::LogVerbatim("PatternRecognitionbyCLUE3D") << "RefLayer: " << layerId << " SoaIDX: " << i;
        edm::LogVerbatim("PatternRecognitionbyCLUE3D") << "NextLayer: " << currentLayer;
      }
      const auto &tileOnLayer = tiles[currentLayer];
      bool onSameLayer = (currentLayer == layerId);
      if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
        edm::LogVerbatim("PatternRecognitionbyCLUE3D") << "onSameLayer: " << onSameLayer;
      }
      const int etaWindow = 2;
      const int phiWindow = 2;
      int etaBinMin = std::max(tileOnLayer.etaBin(clustersOnLayer.eta[i]) - etaWindow, 0);
      int etaBinMax = std::min(tileOnLayer.etaBin(clustersOnLayer.eta[i]) + etaWindow, nEtaBin - 1);
      int phiBinMin = tileOnLayer.phiBin(clustersOnLayer.phi[i]) - phiWindow;
      int phiBinMax = tileOnLayer.phiBin(clustersOnLayer.phi[i]) + phiWindow;
      if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
        edm::LogVerbatim("PatternRecognitionbyCLUE3D") << "eta: " << clustersOnLayer.eta[i];
        edm::LogVerbatim("PatternRecognitionbyCLUE3D") << "phi: " << clustersOnLayer.phi[i];
        edm::LogVerbatim("PatternRecognitionbyCLUE3D") << "etaBinMin: " << etaBinMin << ", etaBinMax: " << etaBinMax;
        edm::LogVerbatim("PatternRecognitionbyCLUE3D") << "phiBinMin: " << phiBinMin << ", phiBinMax: " << phiBinMax;
      }
      for (int ieta = etaBinMin; ieta <= etaBinMax; ++ieta) {
        auto offset = ieta * nPhiBin;
        if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
          edm::LogVerbatim("PatternRecognitionbyCLUE3D") << "offset: " << offset;
        }
        for (int iphi_it = phiBinMin; iphi_it <= phiBinMax; ++iphi_it) {
          int iphi = ((iphi_it % nPhiBin + nPhiBin) % nPhiBin);
          if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
            edm::LogVerbatim("PatternRecognitionbyCLUE3D") << "iphi: " << iphi;
            edm::LogVerbatim("PatternRecognitionbyCLUE3D")
                << "Entries in tileBin: " << tileOnLayer[offset + iphi].size();
          }
          for (auto otherClusterIdx : tileOnLayer[offset + iphi]) {
            auto const &layerandSoa = layerIdx2layerandSoa[otherClusterIdx];
            // Skip masked layer clusters
            if ((layerandSoa.first == -1) && (layerandSoa.second == -1)) {
              if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
                edm::LogVerbatim("PatternRecognitionbyCLUE3D") << "Skipping masked layerIdx " << otherClusterIdx;
              }
              continue;
            }
            auto const &clustersLayer = clusters_[layerandSoa.first];
            if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
              edm::LogVerbatim("PatternRecognitionbyCLUE3D")
                  << "OtherLayer: " << layerandSoa.first << " SoaIDX: " << layerandSoa.second;
              edm::LogVerbatim("PatternRecognitionbyCLUE3D") << "OtherEta: " << clustersLayer.eta[layerandSoa.second];
              edm::LogVerbatim("PatternRecognitionbyCLUE3D") << "OtherPhi: " << clustersLayer.phi[layerandSoa.second];
            }
            bool reachable = false;
            if (useAbsoluteProjectiveScale_) {
              if (useClusterDimensionXY_) {
                reachable = isReachable(clustersOnLayer.r_over_absz[i] * clustersOnLayer.z[i],
                                        clustersLayer.r_over_absz[layerandSoa.second] * clustersOnLayer.z[i],
                                        clustersOnLayer.phi[i],
                                        clustersLayer.phi[layerandSoa.second],
                                        clustersOnLayer.radius[i] * clustersOnLayer.radius[i]);
              } else {
                // Still differentiate between silicon and Scintillator.
                // Silicon has yet to be studied further.
                if (clustersOnLayer.isSilicon[i]) {
                  reachable = isReachable(clustersOnLayer.r_over_absz[i] * clustersOnLayer.z[i],
                                          clustersLayer.r_over_absz[layerandSoa.second] * clustersOnLayer.z[i],
                                          clustersOnLayer.phi[i],
                                          clustersLayer.phi[layerandSoa.second],
                                          densityXYDistanceSqr_);
                } else {
                  reachable = isReachable(clustersOnLayer.r_over_absz[i] * clustersOnLayer.z[i],
                                          clustersLayer.r_over_absz[layerandSoa.second] * clustersOnLayer.z[i],
                                          clustersOnLayer.phi[i],
                                          clustersLayer.phi[layerandSoa.second],
                                          clustersOnLayer.radius[i] * clustersOnLayer.radius[i]);
                }
              }
            } else {
              reachable = (reco::deltaR2(clustersOnLayer.eta[i],
                                         clustersOnLayer.phi[i],
                                         clustersLayer.eta[layerandSoa.second],
                                         clustersLayer.phi[layerandSoa.second]) < densityEtaPhiDistanceSqr_);
            }
            if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
              edm::LogVerbatim("PatternRecognitionbyCLUE3D") << "Distance[eta,phi]: "
                                                             << reco::deltaR2(clustersOnLayer.eta[i],
                                                                              clustersOnLayer.phi[i],
                                                                              clustersLayer.eta[layerandSoa.second],
                                                                              clustersLayer.phi[layerandSoa.second]);
              auto dist = distance_debug(
                  clustersOnLayer.r_over_absz[i],
                  clustersLayer.r_over_absz[layerandSoa.second],
                  clustersOnLayer.r_over_absz[i] * std::abs(clustersOnLayer.phi[i]),
                  clustersLayer.r_over_absz[layerandSoa.second] * std::abs(clustersLayer.phi[layerandSoa.second]));
              edm::LogVerbatim("PatternRecognitionbyCLUE3D") << "Distance[cm]: " << (dist * clustersOnLayer.z[i]);
              edm::LogVerbatim("PatternRecognitionbyCLUE3D")
                  << "Energy Other:   " << clustersLayer.energy[layerandSoa.second];
              edm::LogVerbatim("PatternRecognitionbyCLUE3D") << "Cluster radius: " << clustersOnLayer.radius[i];
            }
            if (reachable) {
              float factor_same_layer_different_cluster = (onSameLayer && !densityOnSameLayer_) ? 0.f : 1.f;
              auto energyToAdd = (clustersOnLayer.layerClusterOriginalIdx[i] == otherClusterIdx
                                      ? 1.f
                                      : kernelDensityFactor_ * factor_same_layer_different_cluster) *
                                 clustersLayer.energy[layerandSoa.second];
              clustersOnLayer.rho[i] += energyToAdd;
              clustersOnLayer.z_extension[i] = deltaLayersZ;
              if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
                edm::LogVerbatim("PatternRecognitionbyCLUE3D")
                    << "Adding " << energyToAdd << " partial " << clustersOnLayer.rho[i];
              }
            }
          }  // end of loop on possible compatible clusters
        }    // end of loop over phi-bin region
      }      // end of loop over eta-bin region
    }        // end of loop on the sibling layers
    if (rescaleDensityByZ_) {
      if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
        edm::LogVerbatim("PatternRecognitionbyCLUE3D")
            << "Rescaling original density: " << clustersOnLayer.rho[i] << " by Z: " << deltaLayersZ
            << " to final density/cm: " << clustersOnLayer.rho[i] / deltaLayersZ;
      }
      clustersOnLayer.rho[i] /= deltaLayersZ;
    }
  }  // end of loop over clusters on this layer
}

template <typename TILES>
void PatternRecognitionbyCLUE3D<TILES>::calculateDistanceToHigher(
    const TILES &tiles, const int layerId, const std::vector<std::pair<int, int>> &layerIdx2layerandSoa) {
  constexpr int nEtaBin = TILES::constants_type_t::nEtaBins;
  constexpr int nPhiBin = TILES::constants_type_t::nPhiBins;
  auto &clustersOnLayer = clusters_[layerId];
  unsigned int numberOfClusters = clustersOnLayer.x.size();

  auto distanceSqr = [](float r0, float r1, float phi0, float phi1) -> float {
    auto delta_phi = reco::deltaPhi(phi0, phi1);
    return (r0 - r1) * (r0 - r1) + r1 * r1 * delta_phi * delta_phi;
  };

  for (unsigned int i = 0; i < numberOfClusters; i++) {
    if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
      edm::LogVerbatim("PatternRecognitionbyCLUE3D")
          << "Starting searching nearestHigher on " << layerId << " with rho: " << clustersOnLayer.rho[i]
          << " at eta, phi: " << tiles[layerId].etaBin(clustersOnLayer.eta[i]) << ", "
          << tiles[layerId].phiBin(clustersOnLayer.phi[i]);
    }
    // We need to partition the two sides of the HGCAL detector
    auto lastLayerPerSide = static_cast<int>(rhtools_.lastLayer(false));
    int minLayer = 0;
    int maxLayer = 2 * lastLayerPerSide - 1;
    if (layerId < lastLayerPerSide) {
      minLayer = std::max(layerId - densitySiblingLayers_, minLayer);
      maxLayer = std::min(layerId + densitySiblingLayers_, lastLayerPerSide - 1);
    } else {
      minLayer = std::max(layerId - densitySiblingLayers_, lastLayerPerSide + 1);
      maxLayer = std::min(layerId + densitySiblingLayers_, maxLayer);
    }
    constexpr float maxDelta = std::numeric_limits<float>::max();
    float i_delta = maxDelta;
    std::pair<int, int> i_nearestHigher(-1, -1);
    std::pair<float, int> nearest_distances(maxDelta, std::numeric_limits<int>::max());
    for (int currentLayer = minLayer; currentLayer <= maxLayer; currentLayer++) {
      if (!nearestHigherOnSameLayer_ && (layerId == currentLayer))
        continue;
      const auto &tileOnLayer = tiles[currentLayer];
      int etaWindow = 1;
      int phiWindow = 1;
      int etaBinMin = std::max(tileOnLayer.etaBin(clustersOnLayer.eta[i]) - etaWindow, 0);
      int etaBinMax = std::min(tileOnLayer.etaBin(clustersOnLayer.eta[i]) + etaWindow, nEtaBin);
      int phiBinMin = tileOnLayer.phiBin(clustersOnLayer.phi[i]) - phiWindow;
      int phiBinMax = tileOnLayer.phiBin(clustersOnLayer.phi[i]) + phiWindow;
      for (int ieta = etaBinMin; ieta <= etaBinMax; ++ieta) {
        auto offset = ieta * nPhiBin;
        for (int iphi_it = phiBinMin; iphi_it <= phiBinMax; ++iphi_it) {
          int iphi = ((iphi_it % nPhiBin + nPhiBin) % nPhiBin);
          if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
            edm::LogVerbatim("PatternRecognitionbyCLUE3D")
                << "Searching nearestHigher on " << currentLayer << " eta, phi: " << ieta << ", " << iphi_it << " "
                << iphi << " " << offset << " " << (offset + iphi);
          }
          for (auto otherClusterIdx : tileOnLayer[offset + iphi]) {
            auto const &layerandSoa = layerIdx2layerandSoa[otherClusterIdx];
            // Skip masked layer clusters
            if ((layerandSoa.first == -1) && (layerandSoa.second == -1))
              continue;
            auto const &clustersOnOtherLayer = clusters_[layerandSoa.first];
            auto dist = maxDelta;
            auto dist_transverse = maxDelta;
            int dist_layers = std::abs(layerandSoa.first - layerId);
            if (useAbsoluteProjectiveScale_) {
              dist_transverse = distanceSqr(clustersOnLayer.r_over_absz[i] * clustersOnLayer.z[i],
                                            clustersOnOtherLayer.r_over_absz[layerandSoa.second] * clustersOnLayer.z[i],
                                            clustersOnLayer.phi[i],
                                            clustersOnOtherLayer.phi[layerandSoa.second]);
              // Add Z-scale to the final distance
              dist = dist_transverse;
            } else {
              dist = reco::deltaR2(clustersOnLayer.eta[i],
                                   clustersOnLayer.phi[i],
                                   clustersOnOtherLayer.eta[layerandSoa.second],
                                   clustersOnOtherLayer.phi[layerandSoa.second]);
              dist_transverse = dist;
            }
            bool foundHigher = (clustersOnOtherLayer.rho[layerandSoa.second] > clustersOnLayer.rho[i]) ||
                               (clustersOnOtherLayer.rho[layerandSoa.second] == clustersOnLayer.rho[i] &&
                                clustersOnOtherLayer.layerClusterOriginalIdx[layerandSoa.second] >
                                    clustersOnLayer.layerClusterOriginalIdx[i]);
            if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
              edm::LogVerbatim("PatternRecognitionbyCLUE3D")
                  << "Searching nearestHigher on " << currentLayer
                  << " with rho: " << clustersOnOtherLayer.rho[layerandSoa.second]
                  << " on layerIdxInSOA: " << layerandSoa.first << ", " << layerandSoa.second
                  << " with distance: " << sqrt(dist) << " foundHigher: " << foundHigher;
            }
            if (foundHigher && dist <= i_delta) {
              // update i_delta
              i_delta = dist;
              nearest_distances = std::make_pair(sqrt(dist_transverse), dist_layers);
              // update i_nearestHigher
              i_nearestHigher = layerandSoa;
            }
          }  // End of loop on clusters
        }    // End of loop on phi bins
      }      // End of loop on eta bins
    }        // End of loop on layers

    bool foundNearestInFiducialVolume = (i_delta != maxDelta);
    if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
      edm::LogVerbatim("PatternRecognitionbyCLUE3D")
          << "i_delta: " << i_delta << " passed: " << foundNearestInFiducialVolume << " " << i_nearestHigher.first
          << " " << i_nearestHigher.second << " distances: " << nearest_distances.first << ", "
          << nearest_distances.second;
    }
    if (foundNearestInFiducialVolume) {
      clustersOnLayer.delta[i] = nearest_distances;
      clustersOnLayer.nearestHigher[i] = i_nearestHigher;
    } else {
      // otherwise delta is guaranteed to be larger outlierDeltaFactor_*delta_c
      // we can safely maximize delta to be maxDelta
      clustersOnLayer.delta[i] = std::make_pair(maxDelta, std::numeric_limits<int>::max());
      clustersOnLayer.nearestHigher[i] = {-1, -1};
    }
  }  // End of loop on clusters
}

template <typename TILES>
int PatternRecognitionbyCLUE3D<TILES>::findAndAssignTracksters(
    const TILES &tiles, const std::vector<std::pair<int, int>> &layerIdx2layerandSoa) {
  unsigned int nTracksters = 0;

  std::vector<std::pair<int, int>> localStack;
  auto critical_transverse_distance = useAbsoluteProjectiveScale_ ? criticalXYDistance_ : criticalEtaPhiDistance_;
  // find cluster seeds and outlier
  for (unsigned int layer = 0; layer < 2 * rhtools_.lastLayer(); layer++) {
    auto &clustersOnLayer = clusters_[layer];
    unsigned int numberOfClusters = clustersOnLayer.x.size();
    for (unsigned int i = 0; i < numberOfClusters; i++) {
      // initialize clusterIndex
      clustersOnLayer.clusterIndex[i] = -1;
      bool isSeed = (clustersOnLayer.delta[i].first > critical_transverse_distance ||
                     clustersOnLayer.delta[i].second > criticalZDistanceLyr_) &&
                    (clustersOnLayer.rho[i] >= criticalDensity_) &&
                    (clustersOnLayer.energy[i] / clustersOnLayer.rho[i] > criticalSelfDensity_);
      if (!clustersOnLayer.isSilicon[i]) {
        isSeed = (clustersOnLayer.delta[i].first > clustersOnLayer.radius[i] ||
                  clustersOnLayer.delta[i].second > criticalZDistanceLyr_) &&
                 (clustersOnLayer.rho[i] >= criticalDensity_) &&
                 (clustersOnLayer.energy[i] / clustersOnLayer.rho[i] > criticalSelfDensity_);
      }
      bool isOutlier = (clustersOnLayer.delta[i].first > outlierMultiplier_ * critical_transverse_distance) &&
                       (clustersOnLayer.rho[i] < criticalDensity_);
      if (isSeed) {
        if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
          edm::LogVerbatim("PatternRecognitionbyCLUE3D")
              << "Found seed on Layer " << layer << " SOAidx: " << i << " assigned ClusterIdx: " << nTracksters;
        }
        clustersOnLayer.clusterIndex[i] = nTracksters++;
        clustersOnLayer.isSeed[i] = true;
        localStack.emplace_back(layer, i);
      } else if (!isOutlier) {
        auto [lyrIdx, soaIdx] = clustersOnLayer.nearestHigher[i];
        if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
          edm::LogVerbatim("PatternRecognitionbyCLUE3D")
              << "Found follower on Layer " << layer << " SOAidx: " << i << " attached to cluster on layer: " << lyrIdx
              << " SOAidx: " << soaIdx;
        }
        if (lyrIdx >= 0)
          clusters_[lyrIdx].followers[soaIdx].emplace_back(layer, i);
      } else {
        if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
          edm::LogVerbatim("PatternRecognitionbyCLUE3D")
              << "Found Outlier on Layer " << layer << " SOAidx: " << i << " with rho: " << clustersOnLayer.rho[i]
              << " and delta: " << clustersOnLayer.delta[i].first << ", " << clustersOnLayer.delta[i].second;
        }
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

template <typename TILES>
void PatternRecognitionbyCLUE3D<TILES>::fillPSetDescription(edm::ParameterSetDescription &iDesc) {
  iDesc.add<int>("algo_verbosity", 0);
  iDesc.add<double>("criticalDensity", 4)->setComment("in GeV");
  iDesc.add<double>("criticalSelfDensity", 0.15 /* roughly 1/(densitySiblingLayers+1) */)
      ->setComment("Minimum ratio of self_energy/local_density to become a seed.");
  iDesc.add<int>("densitySiblingLayers", 3)
      ->setComment(
          "inclusive, layers to consider while computing local density and searching for nearestHigher higher");
  iDesc.add<double>("densityEtaPhiDistanceSqr", 0.0008);
  iDesc.add<double>("densityXYDistanceSqr", 3.24 /*6.76*/)
      ->setComment("in cm, 2.6*2.6, distance on the transverse plane to consider for local density");
  iDesc.add<double>("kernelDensityFactor", 0.2)
      ->setComment("Kernel factor to be applied to other LC while computing the local density");
  iDesc.add<bool>("densityOnSameLayer", false);
  iDesc.add<bool>("nearestHigherOnSameLayer", false)
      ->setComment("Allow the nearestHigher to be located on the same layer");
  iDesc.add<bool>("useAbsoluteProjectiveScale", true)
      ->setComment("Express all cuts in terms of r/z*z_0{,phi} projective variables");
  iDesc.add<bool>("useClusterDimensionXY", false)
      ->setComment(
          "Boolean. If true use the estimated cluster radius to determine the cluster compatibility while computing "
          "the local density");
  iDesc.add<bool>("rescaleDensityByZ", false)
      ->setComment(
          "Rescale local density by the extension of the Z 'volume' explored. The transvere dimension is, at present, "
          "fixed and factored out.");
  iDesc.add<double>("criticalEtaPhiDistance", 0.025)
      ->setComment("Minimal distance in eta,phi space from nearestHigher to become a seed");
  iDesc.add<double>("criticalXYDistance", 1.8)
      ->setComment("Minimal distance in cm on the XY plane from nearestHigher to become a seed");
  iDesc.add<int>("criticalZDistanceLyr", 5)
      ->setComment("Minimal distance in layers along the Z axis from nearestHigher to become a seed");
  iDesc.add<double>("outlierMultiplier", 2);
  iDesc.add<int>("minNumLayerCluster", 2)->setComment("Not Inclusive");
  iDesc.add<std::string>("eid_input_name", "input");
  iDesc.add<std::string>("eid_output_name_energy", "output/regressed_energy");
  iDesc.add<std::string>("eid_output_name_id", "output/id_probabilities");
  iDesc.add<double>("eid_min_cluster_energy", 1.);
  iDesc.add<int>("eid_n_layers", 50);
  iDesc.add<int>("eid_n_clusters", 10);
}

template class ticl::PatternRecognitionbyCLUE3D<TICLLayerTiles>;
template class ticl::PatternRecognitionbyCLUE3D<TICLLayerTilesHFNose>;
