#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalCLUEAlgo.h"

// Geometry
#include "DataFormats/TICL/interface/CaloClusterHostCollection.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/TICL/interface/AssociationMap.h"
#include "DataFormats/TICL/interface/FillAssociator.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
//
#include "DataFormats/CaloRecHit/interface/CaloID.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "CLUEstering/CLUEstering.hpp"
#include "oneapi/tbb.h"
#include "oneapi/tbb/task_arena.h"
#include <limits>

using namespace hgcal_clustering;

template <typename T, typename STRATEGY>
void HGCalCLUEAlgoT<T, STRATEGY>::getEventSetupPerAlgorithm(const edm::EventSetup &es) {
  cells_.clear();
  numberOfClustersPerLayer_.clear();
  cells_.resize(2 * (maxlayer_ + 1));
  numberOfClustersPerLayer_.resize(2 * (maxlayer_ + 1), 0);
}

template <typename T, typename STRATEGY>
void HGCalCLUEAlgoT<T, STRATEGY>::populate(const HGCRecHitCollection &hits) {
  // loop over all hits and create the Hexel structure, skip energies below ecut
  if (dependSensor_) {
    // for each layer and wafer calculate the thresholds (sigmaNoise and energy)
    // once
    computeThreshold();
  }

  for (unsigned int i = 0; i < hits.size(); ++i) {
    const HGCRecHit &hgrh = hits[i];
    DetId detid = hgrh.detid();
    unsigned int layerOnSide = (rhtools_.getLayerWithOffset(detid) - 1);

    // set sigmaNoise default value 1 to use kappa value directly in case of
    // sensor-independent thresholds
    float sigmaNoise = 1.f;
    if (dependSensor_) {
      int thickness_index = rhtools_.getSiThickIndex(detid);
      if (thickness_index == -1)
        thickness_index = maxNumberOfThickIndices_;

      double storedThreshold = thresholds_[layerOnSide][thickness_index];
      if (detid.det() == DetId::HGCalHSi || detid.subdetId() == HGCHEF) {
        storedThreshold = thresholds_[layerOnSide][thickness_index + deltasi_index_regemfac_];
      }
      sigmaNoise = v_sigmaNoise_[layerOnSide][thickness_index];

      if (hgrh.energy() < storedThreshold)
        continue;  // this sets the ZS threshold at ecut times the sigma noise
                   // for the sensor
    }
    if (!dependSensor_ && hgrh.energy() < ecut_)
      continue;
    const GlobalPoint position(rhtools_.getPosition(detid));
    int offset = ((rhtools_.zside(detid) + 1) >> 1) * maxlayer_;
    int layer = layerOnSide + offset;
    // setting the layer position only once per layer
    if (cells_[layer].layerDim3 == std::numeric_limits<float>::infinity())
      cells_[layer].layerDim3 = position.z();

    cells_[layer].detid.emplace_back(detid);
    if constexpr (std::is_same_v<STRATEGY, HGCalScintillatorStrategy>) {
      cells_[layer].dim1.emplace_back(position.eta());
      cells_[layer].dim2.emplace_back(position.phi());
    }  // else, isSilicon == true and eta phi values will not be used
    else {
      cells_[layer].dim1.emplace_back(position.x());
      cells_[layer].dim2.emplace_back(position.y());
    }
    cells_[layer].weight.emplace_back(hgrh.energy());
    cells_[layer].sigmaNoise.emplace_back(sigmaNoise);
  }
}

// Create a vector of Hexels associated to one cluster from a collection of
// HGCalRecHits - this can be used directly to make the final cluster list -
// this method can be invoked multiple times for the same event with different
// input (reset should be called between events)
template <typename T, typename STRATEGY>
void HGCalCLUEAlgoT<T, STRATEGY>::makeClusters() {
  for (auto l = 0u; l < maxlayer_; ++l) {
    float delta;
    if constexpr (std::is_same_v<STRATEGY, HGCalSiliconStrategy>) {
      // maximum search distance (critical distance) for local density
      // calculation
      float delta_c;
      if (l % maxlayer_ < lastLayerEE_)
        delta_c = vecDeltas_[0];
      else if (l % maxlayer_ < (firstLayerBH_ - 1))
        delta_c = vecDeltas_[1];
      else
        delta_c = vecDeltas_[2];
      delta = delta_c;
    } else {
      float delta_r = vecDeltas_[3];
      delta = delta_r;
    }

    auto clusterer = clue::Clusterer<2>(delta, kappa_);
    auto queue = clue::get_queue(0u);
    auto points = clue::PointsHost<2>(
        queue, cells_[l].dim1.size(), cells_[l].dim1, cells_[l].dim2, cells_[l].weight, cells_[l].clusterIndex);
    points.set_density_uncertainty(cells_[l].sigmaNoise);
    clusterer.make_clusters(points);
    numberOfClustersPerLayer_[l] = points.n_clusters();
  }
#if DEBUG_CLUSTERS_ALPAKA
  hgcalUtils::DumpLegacySoA dumperLegacySoA;
  dumperLegacySoA.dumpInfos(cells_, moduleType_);
#endif
}

template <typename T, typename STRATEGY>
std::vector<reco::BasicCluster> HGCalCLUEAlgoT<T, STRATEGY>::getClustersLegacy(bool) {
  return std::vector<reco::BasicCluster>(1);
}

template <typename T, typename STRATEGY>
ticl::LayerClustersAndAssociations HGCalCLUEAlgoT<T, STRATEGY>::getClusters(bool) {
  std::vector<int> offsets(numberOfClustersPerLayer_.size(), 0);
  int maxClustersOnLayer = numberOfClustersPerLayer_[0];
  for (unsigned layerId = 1; layerId < offsets.size(); ++layerId) {
    offsets[layerId] = offsets[layerId - 1] + numberOfClustersPerLayer_[layerId - 1];
    maxClustersOnLayer = std::max(maxClustersOnLayer, numberOfClustersPerLayer_[layerId]);
  }
  auto totalNumberOfClusters = offsets.back() + numberOfClustersPerLayer_.back();

  // std::vector<std::vector<int>> cellsIdInCluster;
  // cellsIdInCluster.reserve(maxClustersOnLayer);
  const auto total_rechits = std::accumulate(
      cells_.begin(), cells_.end(), 0, [](auto acc, const auto &cell) { return acc + cell.dim1.size(); });
  ticl::LayerClustersAndAssociations clusters_and_associations(totalNumberOfClusters, total_rechits);

  std::vector<ticl::HitAndFraction> detid_and_fractions;
  std::vector<int> cluster_hit_associations;
  for (unsigned int layerId = 0; layerId < 2 * maxlayer_ + 2; ++layerId) {
    auto queue = clue::get_queue(0u);
    auto points = clue::PointsHost<2>(queue,
                                      cells_[layerId].dim1.size(),
                                      cells_[layerId].dim1,
                                      cells_[layerId].dim2,
                                      cells_[layerId].weight,
                                      cells_[layerId].clusterIndex);

    std::ranges::copy(points.clusterIndexes(), std::back_inserter(cluster_hit_associations));

    auto clusters = clue::get_clusters(points);
    auto to_hit_and_fraction = [&](auto idx) { return ticl::HitAndFraction{cells_[layerId].detid[idx], -1.f}; };
    std::ranges::copy(clusters | std::views::transform(to_hit_and_fraction), std::back_inserter(detid_and_fractions));
    for (auto cl = 0u; cl < clusters.size(); ++cl) {
      const auto cluster = clusters[cl];
      auto x = 0.f;
      auto y = 0.f;
      const auto z = cells_[layerId].layerDim3;
      auto energy = std::reduce(
          cluster.begin(), cluster.end(), 0.f, [&](auto acc, auto idx) { return acc + points.weights()[idx]; });
      auto max_energy_it = std::ranges::max_element(points.weights());
      const auto max_energy_idx = std::distance(points.weights().begin(), max_energy_it);
      const auto max_energy_detid = cells_[layerId].detid[max_energy_idx];

      if constexpr (std::is_same_v<STRATEGY, HGCalSiliconStrategy>) {
        auto thick = rhtools_.getSiThickIndex(max_energy_detid);
        auto total_weight_log = 0.f;
        for (auto p : cluster) {
          const auto d1 = points.coords(0)[p] - points.coords(0)[max_energy_idx];
          const auto d2 = points.coords(1)[p] - points.coords(1)[max_energy_idx];
          if ((d1 * d1 + d2 * d2) < positionDeltaRho2_) {
            auto Wi = std::max(thresholdW0_[thick] + std::log(points.weights()[p] / energy), 0.);
            x += points.coords(0)[p] * Wi;
            y += points.coords(1)[p] * Wi;
            total_weight_log += Wi;
          }
        }

        if (total_weight_log != 0.) {
          auto inv_tot_weight = 1.f / total_weight_log;
          x *= inv_tot_weight;
          y *= inv_tot_weight;
        } else {
          x = points.coords(0)[max_energy_idx];
          y = points.coords(1)[max_energy_idx];
        }
      } else {
        const auto centroid = clue::weighted_cluster_centroid(points, cl);
        x = centroid[0];
        y = centroid[1];
      }

      auto globalClusterIndex = cl + offsets[layerId];
      auto &layer_clusters_view = clusters_and_associations.layer_clusters->view();
      layer_clusters_view.position().x()[globalClusterIndex] = x;
      layer_clusters_view.position().y()[globalClusterIndex] = y;
      layer_clusters_view.position().z()[globalClusterIndex] = z;
      layer_clusters_view.position().cells()[globalClusterIndex] = clusters.count(cl);
      layer_clusters_view.energy().energy()[globalClusterIndex] = energy;
      layer_clusters_view.energy().correctedEnergy()[globalClusterIndex] = -1.f;
      layer_clusters_view.energy().correctedEnergyUncertainty()[globalClusterIndex] = -1.f;
      layer_clusters_view.indexes().caloID()[globalClusterIndex] = reco::CaloID::DET_HGCAL_ENDCAP;
      layer_clusters_view.indexes().algoID()[globalClusterIndex] = algoId_;
      // TODO: do we really care about the seed?
      // layer_clusters.view().indexes().seedID()[globalClusterIndex] = seedDetId;
      layer_clusters_view.indexes().flags()[globalClusterIndex] = 0;
    }
  }
  alpaka_serial_sync::Queue queue(cms::alpakatools::host());
  ticl::associator::fill<alpaka_serial_sync::Acc1D>(
      queue,
      clusters_and_associations.hits_and_fractions->view(),
      static_cast<std::span<const int>>(cluster_hit_associations),
      static_cast<std::span<const ticl::HitAndFraction>>(detid_and_fractions));

  return clusters_and_associations;
}

template <typename T, typename STRATEGY>
void HGCalCLUEAlgoT<T, STRATEGY>::computeThreshold() {
  // To support the TDR geometry and also the post-TDR one (v9 onwards), we
  // need to change the logic of the vectors containing signal to noise and
  // thresholds. The first 3 indices will keep on addressing the different
  // thicknesses of the Silicon detectors in CE_E , the next 3 indices will
  // address the thicknesses of the Silicon detectors in CE_H, while the last
  // one, number 6 (the seventh) will address the Scintillators. This change
  // will support both geometries at the same time.

  if (initialized_)
    return;  // only need to calculate thresholds once

  initialized_ = true;

  std::vector<double> dummy;

  dummy.resize(maxNumberOfThickIndices_ + !isNose_,
               0);  // +1 to accomodate for the Scintillators
  thresholds_.resize(maxlayer_, dummy);
  v_sigmaNoise_.resize(maxlayer_, dummy);

  for (unsigned ilayer = 1; ilayer <= maxlayer_; ++ilayer) {
    for (unsigned ithick = 0; ithick < maxNumberOfThickIndices_; ++ithick) {
      float sigmaNoise = 0.001f * fcPerEle_ * nonAgedNoises_[ithick] * dEdXweights_[ilayer] /
                         (fcPerMip_[ithick] * thicknessCorrection_[ithick]);
      thresholds_[ilayer - 1][ithick] = sigmaNoise * ecut_;
      v_sigmaNoise_[ilayer - 1][ithick] = sigmaNoise;
      LogDebug("HGCalCLUEAlgo") << "ilayer: " << ilayer << " nonAgedNoises: " << nonAgedNoises_[ithick]
                                << " fcPerEle: " << fcPerEle_ << " fcPerMip: " << fcPerMip_[ithick]
                                << " noiseMip: " << fcPerEle_ * nonAgedNoises_[ithick] / fcPerMip_[ithick]
                                << " sigmaNoise: " << sigmaNoise << "\n";
    }

    if (!isNose_) {
      float scintillators_sigmaNoise = 0.001f * noiseMip_ * dEdXweights_[ilayer] / sciThicknessCorrection_;
      thresholds_[ilayer - 1][maxNumberOfThickIndices_] = ecut_ * scintillators_sigmaNoise;
      v_sigmaNoise_[ilayer - 1][maxNumberOfThickIndices_] = scintillators_sigmaNoise;
      LogDebug("HGCalCLUEAlgo") << "ilayer: " << ilayer << " noiseMip: " << noiseMip_
                                << " scintillators_sigmaNoise: " << scintillators_sigmaNoise << "\n";
    }
  }
}

// explicit template instantiation
template class HGCalCLUEAlgoT<HGCalSiliconLayerTiles, HGCalSiliconStrategy>;
template class HGCalCLUEAlgoT<HGCalScintillatorLayerTiles, HGCalScintillatorStrategy>;
template class HGCalCLUEAlgoT<HFNoseLayerTiles, HGCalSiliconStrategy>;
