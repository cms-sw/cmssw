// Author: Felice Pantaleo, Marco Rovere - felice.pantaleo@cern.ch, marco.rovere@cern.ch
// Date: 11/2018
#include <algorithm>
#include <set>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "PatternRecognitionbyCA.h"

#include "TrackstersPCA.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/Framework/interface/EventSetup.h"

using namespace ticl;

template <typename TILES>
PatternRecognitionbyCA<TILES>::PatternRecognitionbyCA(const edm::ParameterSet &conf, edm::ConsumesCollector iC)
    : PatternRecognitionAlgoBaseT<TILES>(conf, iC),
      caloGeomToken_(iC.esConsumes<CaloGeometry, CaloGeometryRecord>()),
      theGraph_(std::make_unique<HGCGraphT<TILES>>()),
      oneTracksterPerTrackSeed_(conf.getParameter<bool>("oneTracksterPerTrackSeed")),
      promoteEmptyRegionToTrackster_(conf.getParameter<bool>("promoteEmptyRegionToTrackster")),
      out_in_dfs_(conf.getParameter<bool>("out_in_dfs")),
      max_out_in_hops_(conf.getParameter<int>("max_out_in_hops")),
      min_cos_theta_(conf.getParameter<double>("min_cos_theta")),
      min_cos_pointing_(conf.getParameter<double>("min_cos_pointing")),
      root_doublet_max_distance_from_seed_squared_(
          conf.getParameter<double>("root_doublet_max_distance_from_seed_squared")),
      etaLimitIncreaseWindow_(conf.getParameter<double>("etaLimitIncreaseWindow")),
      skip_layers_(conf.getParameter<int>("skip_layers")),
      max_missing_layers_in_trackster_(conf.getParameter<int>("max_missing_layers_in_trackster")),
      check_missing_layers_(max_missing_layers_in_trackster_ < 100),
      shower_start_max_layer_(conf.getParameter<int>("shower_start_max_layer")),
      min_layers_per_trackster_(conf.getParameter<int>("min_layers_per_trackster")),
      filter_on_categories_(conf.getParameter<std::vector<int>>("filter_on_categories")),
      pid_threshold_(conf.getParameter<double>("pid_threshold")),
      energy_em_over_total_threshold_(conf.getParameter<double>("energy_em_over_total_threshold")),
      max_longitudinal_sigmaPCA_(conf.getParameter<double>("max_longitudinal_sigmaPCA")),
      min_clusters_per_ntuplet_(min_layers_per_trackster_),
      max_delta_time_(conf.getParameter<double>("max_delta_time")),
      computeLocalTime_(conf.getParameter<bool>("computeLocalTime")),
      siblings_maxRSquared_(conf.getParameter<std::vector<double>>("siblings_maxRSquared")){};

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

  edm::EventSetup const &es = input.es;
  const CaloGeometry &geom = es.getData(caloGeomToken_);
  rhtools_.setGeometry(geom);

  theGraph_->setVerbosity(PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_);
  theGraph_->clear();
  if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::None) {
    LogDebug("HGCPatternRecoByCA") << "Making Tracksters with CA" << std::endl;
  }

  constexpr auto isHFnose = std::is_same<TILES, TICLLayerTilesHFNose>::value;
  constexpr int nEtaBin = TILES::constants_type_t::nEtaBins;
  constexpr int nPhiBin = TILES::constants_type_t::nPhiBins;

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
                                    root_doublet_max_distance_from_seed_squared_,
                                    etaLimitIncreaseWindow_,
                                    skip_layers_,
                                    rhtools_.lastLayer(isHFnose),
                                    max_delta_time_,
                                    rhtools_.lastLayerEE(isHFnose),
                                    rhtools_.lastLayerFH(),
                                    siblings_maxRSquared_);

  theGraph_->findNtuplets(foundNtuplets, seedIndices, min_clusters_per_ntuplet_, out_in_dfs_, max_out_in_hops_);
  //#ifdef FP_DEBUG
  const auto &doublets = theGraph_->getAllDoublets();
  int tracksterId = -1;

  // container for holding tracksters before selection
  result.reserve(foundNtuplets.size());

  for (auto const &ntuplet : foundNtuplets) {
    tracksterId++;

    std::set<unsigned int> effective_cluster_idx;

    for (auto const &doublet : ntuplet) {
      auto innerCluster = doublets[doublet].innerClusterId();
      auto outerCluster = doublets[doublet].outerClusterId();

      effective_cluster_idx.insert(innerCluster);
      effective_cluster_idx.insert(outerCluster);

      if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
        LogDebug("HGCPatternRecoByCA") << " New doublet " << doublet << " for trackster: " << result.size()
                                       << " InnerCl " << innerCluster << " " << input.layerClusters[innerCluster].x()
                                       << " " << input.layerClusters[innerCluster].y() << " "
                                       << input.layerClusters[innerCluster].z() << " OuterCl " << outerCluster << " "
                                       << input.layerClusters[outerCluster].x() << " "
                                       << input.layerClusters[outerCluster].y() << " "
                                       << input.layerClusters[outerCluster].z() << " " << tracksterId << std::endl;
      }
    }
    unsigned showerMinLayerId = 99999;
    std::vector<unsigned int> uniqueLayerIds;
    uniqueLayerIds.reserve(effective_cluster_idx.size());
    std::vector<std::pair<unsigned int, unsigned int>> lcIdAndLayer;
    lcIdAndLayer.reserve(effective_cluster_idx.size());
    for (auto const i : effective_cluster_idx) {
      auto const &haf = input.layerClusters[i].hitsAndFractions();
      auto layerId = rhtools_.getLayerWithOffset(haf[0].first);
      showerMinLayerId = std::min(layerId, showerMinLayerId);
      uniqueLayerIds.push_back(layerId);
      lcIdAndLayer.emplace_back(i, layerId);
    }
    std::sort(uniqueLayerIds.begin(), uniqueLayerIds.end());
    uniqueLayerIds.erase(std::unique(uniqueLayerIds.begin(), uniqueLayerIds.end()), uniqueLayerIds.end());
    unsigned int numberOfLayersInTrackster = uniqueLayerIds.size();
    if (check_missing_layers_) {
      int numberOfMissingLayers = 0;
      unsigned int j = showerMinLayerId;
      unsigned int indexInVec = 0;
      for (const auto &layer : uniqueLayerIds) {
        if (layer != j) {
          numberOfMissingLayers++;
          j++;
          if (numberOfMissingLayers > max_missing_layers_in_trackster_) {
            numberOfLayersInTrackster = indexInVec;
            for (auto &llpair : lcIdAndLayer) {
              if (llpair.second >= layer) {
                effective_cluster_idx.erase(llpair.first);
              }
            }
            break;
          }
        }
        indexInVec++;
        j++;
      }
    }
    if ((numberOfLayersInTrackster >= min_layers_per_trackster_) and (showerMinLayerId <= shower_start_max_layer_)) {
      // Put back indices, in the form of a Trackster, into the results vector
      Trackster tmp;
      tmp.vertices().reserve(effective_cluster_idx.size());
      tmp.vertex_multiplicity().resize(effective_cluster_idx.size(), 1);
      //regions and seedIndices can have different size
      //if a seeding region does not lead to any trackster
      tmp.setSeed(input.regions[0].collectionID, seedIndices[tracksterId]);

      std::copy(std::begin(effective_cluster_idx), std::end(effective_cluster_idx), std::back_inserter(tmp.vertices()));
      result.push_back(tmp);
    }
  }
  ticl::assignPCAtoTracksters(result,
                              input.layerClusters,
                              input.layerClustersTime,
                              rhtools_.getPositionLayer(rhtools_.lastLayerEE(isHFnose), isHFnose).z(),
                              rhtools_,
                              computeLocalTime_);

  theGraph_->clear();
}

template <typename TILES>
void PatternRecognitionbyCA<TILES>::filter(std::vector<Trackster> &output,
                                           const std::vector<Trackster> &inTracksters,
                                           const typename PatternRecognitionAlgoBaseT<TILES>::Inputs &input,
                                           std::unordered_map<int, std::vector<int>> &seedToTracksterAssociation) {
  auto filter_on_pids = [&](const ticl::Trackster &t) -> bool {
    auto cumulative_prob = 0.;
    for (auto index : filter_on_categories_) {
      cumulative_prob += t.id_probabilities(index);
    }
    return (cumulative_prob <= pid_threshold_) &&
           (t.raw_em_energy() < energy_em_over_total_threshold_ * t.raw_energy());
  };

  std::vector<unsigned int> selectedTrackstersIds;
  for (unsigned i = 0; i < inTracksters.size(); ++i) {
    auto &t = inTracksters[i];
    if (!filter_on_pids(t) and t.sigmasPCA()[0] < max_longitudinal_sigmaPCA_) {
      selectedTrackstersIds.push_back(i);
    }
  }
  output.reserve(selectedTrackstersIds.size());
  bool isRegionalIter = !input.regions.empty() && (input.regions[0].index != -1);
  for (unsigned i = 0; i < selectedTrackstersIds.size(); ++i) {
    const auto &t = inTracksters[selectedTrackstersIds[i]];
    if (isRegionalIter) {
      seedToTracksterAssociation[t.seedIndex()].push_back(i);
    }
    output.push_back(t);
  }

  // Now decide if the tracksters from the track-based iterations have to be merged
  if (oneTracksterPerTrackSeed_) {
    std::vector<Trackster> tmp;
    mergeTrackstersTRK(output, input.layerClusters, tmp, seedToTracksterAssociation);
    tmp.swap(output);
  }

  // now adding dummy tracksters from seeds not connected to any shower in the result collection
  // these are marked as charged hadrons with probability 1.
  if (promoteEmptyRegionToTrackster_) {
    emptyTrackstersFromSeedsTRK(output, seedToTracksterAssociation, input.regions[0].collectionID);
  }

  if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
    for (auto &trackster : output) {
      LogDebug("HGCPatternRecoByCA") << "Trackster characteristics: " << std::endl;
      LogDebug("HGCPatternRecoByCA") << "Size: " << trackster.vertices().size() << std::endl;
      auto counter = 0;
      for (auto const &p : trackster.id_probabilities()) {
        LogDebug("HGCPatternRecoByCA") << counter++ << ": " << p << std::endl;
      }
    }
  }
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
        if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Basic) {
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

      // Find duplicate LCs
      auto &orig_vtx = outTrackster.vertices();
      auto vtx_sorted{orig_vtx};
      std::sort(std::begin(vtx_sorted), std::end(vtx_sorted));
      for (unsigned int iLC = 1; iLC < vtx_sorted.size(); ++iLC) {
        if (vtx_sorted[iLC] == vtx_sorted[iLC - 1]) {
          // Clean up duplicate LCs
          const auto lcIdx = vtx_sorted[iLC];
          const auto firstEl = std::find(orig_vtx.begin(), orig_vtx.end(), lcIdx);
          const auto firstPos = std::distance(std::begin(orig_vtx), firstEl);
          auto iDup = std::find(std::next(firstEl), orig_vtx.end(), lcIdx);
          while (iDup != orig_vtx.end()) {
            orig_vtx.erase(iDup);
            outTrackster.vertex_multiplicity().erase(outTrackster.vertex_multiplicity().begin() +
                                                     std::distance(std::begin(orig_vtx), iDup));
            outTrackster.vertex_multiplicity()[firstPos] -= 1;
            iDup = std::find(std::next(firstEl), orig_vtx.end(), lcIdx);
          };
        }
      }
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
void PatternRecognitionbyCA<TILES>::fillPSetDescription(edm::ParameterSetDescription &iDesc) {
  iDesc.add<int>("algo_verbosity", 0);
  iDesc.add<bool>("oneTracksterPerTrackSeed", false);
  iDesc.add<bool>("promoteEmptyRegionToTrackster", false);
  iDesc.add<bool>("out_in_dfs", true);
  iDesc.add<int>("max_out_in_hops", 10);
  iDesc.add<double>("min_cos_theta", 0.915);
  iDesc.add<double>("min_cos_pointing", -1.);
  iDesc.add<double>("root_doublet_max_distance_from_seed_squared", 9999);
  iDesc.add<double>("etaLimitIncreaseWindow", 2.1);
  iDesc.add<int>("skip_layers", 0);
  iDesc.add<int>("max_missing_layers_in_trackster", 9999);
  iDesc.add<int>("shower_start_max_layer", 9999)->setComment("make default such that no filtering is applied");
  iDesc.add<int>("min_layers_per_trackster", 10);
  iDesc.add<std::vector<int>>("filter_on_categories", {0});
  iDesc.add<double>("pid_threshold", 0.)->setComment("make default such that no filtering is applied");
  iDesc.add<double>("energy_em_over_total_threshold", -1.)
      ->setComment("make default such that no filtering is applied");
  iDesc.add<double>("max_longitudinal_sigmaPCA", 9999);
  iDesc.add<double>("max_delta_time", 3.)->setComment("nsigma");
  iDesc.add<bool>("computeLocalTime", false);
  iDesc.add<std::vector<double>>("siblings_maxRSquared", {6e-4, 6e-4, 6e-4});
}

template class ticl::PatternRecognitionbyCA<TICLLayerTiles>;
template class ticl::PatternRecognitionbyCA<TICLLayerTilesHFNose>;
