#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalCLUEAlgo.h"

// Geometry
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
//
#include "DataFormats/CaloRecHit/interface/CaloID.h"
#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb.h"
#include <limits>
#include "DataFormats/DetId/interface/DetId.h"

using namespace hgcal_clustering;

template <typename T, typename STRATEGY>
void HGCalCLUEAlgoT<T, STRATEGY>::getEventSetupPerAlgorithm(const edm::EventSetup& es) {
  cells_.clear();
  numberOfClustersPerLayer_.clear();
  cells_.resize(2 * (maxlayer_ + 1));
  numberOfClustersPerLayer_.resize(2 * (maxlayer_ + 1), 0);
}

template <typename T, typename STRATEGY>
void HGCalCLUEAlgoT<T, STRATEGY>::populate(const HGCRecHitCollection& hits) {
  // loop over all hits and create the Hexel structure, skip energies below ecut
  if (dependSensor_) {
    // for each layer and wafer calculate the thresholds (sigmaNoise and energy)
    // once
    computeThreshold();
  }

  for (unsigned int i = 0; i < hits.size(); ++i) {
    const HGCRecHit& hgrh = hits[i];
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

template <typename T, typename STRATEGY>
void HGCalCLUEAlgoT<T, STRATEGY>::prepareDataStructures(unsigned int l) {
  auto cellsSize = cells_[l].detid.size();
  cells_[l].rho.resize(cellsSize, 0.f);
  cells_[l].delta.resize(cellsSize, 9999999);
  cells_[l].nearestHigher.resize(cellsSize, -1);
  cells_[l].clusterIndex.resize(cellsSize, -1);
  cells_[l].followers.resize(cellsSize);
  cells_[l].isSeed.resize(cellsSize, false);
}

// Create a vector of Hexels associated to one cluster from a collection of
// HGCalRecHits - this can be used directly to make the final cluster list -
// this method can be invoked multiple times for the same event with different
// input (reset should be called between events)
template <typename T, typename STRATEGY>
void HGCalCLUEAlgoT<T, STRATEGY>::makeClusters() {
  // assign all hits in each layer to a cluster core
  tbb::this_task_arena::isolate([&] {
    tbb::parallel_for(size_t(0), size_t(2 * maxlayer_ + 2), [&](size_t i) {
      prepareDataStructures(i);
      T lt;
      lt.clear();
      lt.fill(cells_[i].dim1, cells_[i].dim2);

      float delta_c;  // maximum search distance (critical distance) for local
      // density calculation
      if (i % maxlayer_ < lastLayerEE_)
        delta_c = vecDeltas_[0];
      else if (i % maxlayer_ < (firstLayerBH_ - 1))
        delta_c = vecDeltas_[1];
      else
        delta_c = vecDeltas_[2];
      float delta_r = vecDeltas_[3];

      float delta;

      if constexpr (std::is_same_v<STRATEGY, HGCalSiliconStrategy>)
        delta = delta_c;
      else
        delta = delta_r;
      LogDebug("HGCalCLUEAlgo") << "maxlayer: " << maxlayer_ << " lastLayerEE: " << lastLayerEE_
                                << " firstLayerBH: " << firstLayerBH_ << "\n";

      calculateLocalDensity(lt, i, delta);
      calculateDistanceToHigher(lt, i, delta);
      numberOfClustersPerLayer_[i] = findAndAssignClusters(i, delta);
    });
  });
}

template <typename T, typename STRATEGY>
std::vector<reco::BasicCluster> HGCalCLUEAlgoT<T, STRATEGY>::getClusters(bool) {
  std::vector<int> offsets(numberOfClustersPerLayer_.size(), 0);

  int maxClustersOnLayer = numberOfClustersPerLayer_[0];

  for (unsigned layerId = 1; layerId < offsets.size(); ++layerId) {
    offsets[layerId] = offsets[layerId - 1] + numberOfClustersPerLayer_[layerId - 1];

    maxClustersOnLayer = std::max(maxClustersOnLayer, numberOfClustersPerLayer_[layerId]);
  }

  auto totalNumberOfClusters = offsets.back() + numberOfClustersPerLayer_.back();
  clusters_v_.resize(totalNumberOfClusters);
  std::vector<std::vector<int>> cellsIdInCluster;
  cellsIdInCluster.reserve(maxClustersOnLayer);

  for (unsigned int layerId = 0; layerId < 2 * maxlayer_ + 2; ++layerId) {
    cellsIdInCluster.resize(numberOfClustersPerLayer_[layerId]);
    auto& cellsOnLayer = cells_[layerId];
    unsigned int numberOfCells = cellsOnLayer.detid.size();
    auto firstClusterIdx = offsets[layerId];

    for (unsigned int i = 0; i < numberOfCells; ++i) {
      auto clusterIndex = cellsOnLayer.clusterIndex[i];
      if (clusterIndex != -1)
        cellsIdInCluster[clusterIndex].push_back(i);
    }

    std::vector<std::pair<DetId, float>> thisCluster;

    for (auto& cl : cellsIdInCluster) {
      math::XYZPoint position = math::XYZPoint(0.f, 0.f, 0.f);
      float energy = 0.f;
      int seedDetId = -1;

      for (auto cellIdx : cl) {
        energy += cellsOnLayer.weight[cellIdx];
        thisCluster.emplace_back(cellsOnLayer.detid[cellIdx], 1.f);
        if (cellsOnLayer.isSeed[cellIdx]) {
          seedDetId = cellsOnLayer.detid[cellIdx];
        }
      }
      auto globalClusterIndex = cellsOnLayer.clusterIndex[cl[0]] + firstClusterIdx;

      clusters_v_[globalClusterIndex] =
          reco::BasicCluster(energy, position, reco::CaloID::DET_HGCAL_ENDCAP, thisCluster, algoId_);
      clusters_v_[globalClusterIndex].setSeed(seedDetId);
      thisCluster.clear();
    }

    cellsIdInCluster.clear();
  }
  return clusters_v_;
}
template <typename T, typename STRATEGY>
void HGCalCLUEAlgoT<T, STRATEGY>::calculateLocalDensity(const T& lt,
                                                        const unsigned int layerId,
                                                        float delta,
                                                        HGCalSiliconStrategy strategy) {
  auto& cellsOnLayer = cells_[layerId];
  unsigned int numberOfCells = cellsOnLayer.detid.size();
  for (unsigned int i = 0; i < numberOfCells; i++) {
    std::array<int, 4> search_box = lt.searchBox(cellsOnLayer.dim1[i] - delta,
                                                 cellsOnLayer.dim1[i] + delta,
                                                 cellsOnLayer.dim2[i] - delta,
                                                 cellsOnLayer.dim2[i] + delta);

    for (int xBin = search_box[0]; xBin < search_box[1] + 1; ++xBin) {
      for (int yBin = search_box[2]; yBin < search_box[3] + 1; ++yBin) {
        int binId = lt.getGlobalBinByBin(xBin, yBin);
        size_t binSize = lt[binId].size();

        for (unsigned int j = 0; j < binSize; j++) {
          unsigned int otherId = lt[binId][j];
          if (distance(lt, i, otherId, layerId) < delta) {
            cellsOnLayer.rho[i] += (i == otherId ? 1.f : 0.5f) * cellsOnLayer.weight[otherId];
          }
        }
      }
    }
    LogDebug("HGCalCLUEAlgo") << "Debugging calculateLocalDensity: \n"
                              << "  cell: " << i << " eta: " << cellsOnLayer.dim1[i] << " phi: " << cellsOnLayer.dim2[i]
                              << " energy: " << cellsOnLayer.weight[i] << " density: " << cellsOnLayer.rho[i] << "\n";
  }
}
template <typename T, typename STRATEGY>
void HGCalCLUEAlgoT<T, STRATEGY>::calculateLocalDensity(const T& lt,
                                                        const unsigned int layerId,
                                                        float delta,
                                                        HGCalScintillatorStrategy strategy) {
  auto& cellsOnLayer = cells_[layerId];
  unsigned int numberOfCells = cellsOnLayer.detid.size();
  for (unsigned int i = 0; i < numberOfCells; i++) {
    std::array<int, 4> search_box = lt.searchBox(cellsOnLayer.dim1[i] - delta,
                                                 cellsOnLayer.dim1[i] + delta,
                                                 cellsOnLayer.dim2[i] - delta,
                                                 cellsOnLayer.dim2[i] + delta);
    cellsOnLayer.rho[i] += cellsOnLayer.weight[i];
    float northeast(0), northwest(0), southeast(0), southwest(0), all(0);
    for (int etaBin = search_box[0]; etaBin < search_box[1] + 1; ++etaBin) {
      for (int phiBin = search_box[2]; phiBin < search_box[3] + 1; ++phiBin) {
        int phi = (phiBin % T::type::nRows);
        int binId = lt.getGlobalBinByBin(etaBin, phi);
        size_t binSize = lt[binId].size();
        for (unsigned int j = 0; j < binSize; j++) {
          unsigned int otherId = lt[binId][j];
          if (distance(lt, i, otherId, layerId) < delta) {
            int iPhi = HGCScintillatorDetId(cellsOnLayer.detid[i]).iphi();
            int otherIPhi = HGCScintillatorDetId(cellsOnLayer.detid[otherId]).iphi();
            int iEta = HGCScintillatorDetId(cellsOnLayer.detid[i]).ieta();
            int otherIEta = HGCScintillatorDetId(cellsOnLayer.detid[otherId]).ieta();
            int dIPhi = otherIPhi - iPhi;
            dIPhi += abs(dIPhi) < 2 ? 0
                     : dIPhi < 0    ? scintMaxIphi_
                                    : -scintMaxIphi_;  // cells with iPhi=288 and iPhi=1 should be neiboring cells
            int dIEta = otherIEta - iEta;
            LogDebug("HGCalCLUEAlgo") << "  Debugging calculateLocalDensity for Scintillator: \n"
                                      << "    cell: " << otherId << " energy: " << cellsOnLayer.weight[otherId]
                                      << " otherIPhi: " << otherIPhi << " iPhi: " << iPhi << " otherIEta: " << otherIEta
                                      << " iEta: " << iEta << "\n";

            if (otherId != i) {
              auto neighborCellContribution = 0.5f * cellsOnLayer.weight[otherId];
              all += neighborCellContribution;
              if (dIPhi >= 0 && dIEta >= 0)
                northeast += neighborCellContribution;
              if (dIPhi <= 0 && dIEta >= 0)
                southeast += neighborCellContribution;
              if (dIPhi >= 0 && dIEta <= 0)
                northwest += neighborCellContribution;
              if (dIPhi <= 0 && dIEta <= 0)
                southwest += neighborCellContribution;
            }
            LogDebug("HGCalCLUEAlgo") << "  Debugging calculateLocalDensity for Scintillator: \n"
                                      << "    northeast: " << northeast << " southeast: " << southeast
                                      << " northwest: " << northwest << " southwest: " << southwest << "\n";
          }
        }
      }
    }
    float neighborsval = (std::max(northeast, northwest) > std::max(southeast, southwest))
                             ? std::max(northeast, northwest)
                             : std::max(southeast, southwest);
    if (use2x2_)
      cellsOnLayer.rho[i] += neighborsval;
    else
      cellsOnLayer.rho[i] += all;
    LogDebug("HGCalCLUEAlgo") << "Debugging calculateLocalDensity: \n"
                              << "  cell: " << i << " eta: " << cellsOnLayer.dim1[i] << " phi: " << cellsOnLayer.dim2[i]
                              << " energy: " << cellsOnLayer.weight[i] << " density: " << cellsOnLayer.rho[i] << "\n";
  }
}
template <typename T, typename STRATEGY>
void HGCalCLUEAlgoT<T, STRATEGY>::calculateLocalDensity(const T& lt, const unsigned int layerId, float delta) {
  if constexpr (std::is_same_v<STRATEGY, HGCalSiliconStrategy>) {
    calculateLocalDensity(lt, layerId, delta, HGCalSiliconStrategy());
  } else {
    calculateLocalDensity(lt, layerId, delta, HGCalScintillatorStrategy());
  }
}

template <typename T, typename STRATEGY>
void HGCalCLUEAlgoT<T, STRATEGY>::calculateDistanceToHigher(const T& lt, const unsigned int layerId, float delta) {
  auto& cellsOnLayer = cells_[layerId];
  unsigned int numberOfCells = cellsOnLayer.detid.size();

  for (unsigned int i = 0; i < numberOfCells; i++) {
    // initialize delta and nearest higher for i
    float maxDelta = std::numeric_limits<float>::max();
    float i_delta = maxDelta;
    int i_nearestHigher = -1;
    auto range = outlierDeltaFactor_ * delta;
    std::array<int, 4> search_box = lt.searchBox(cellsOnLayer.dim1[i] - range,
                                                 cellsOnLayer.dim1[i] + range,
                                                 cellsOnLayer.dim2[i] - range,
                                                 cellsOnLayer.dim2[i] + range);
    // loop over all bins in the search box
    for (int dim1Bin = search_box[0]; dim1Bin < search_box[1] + 1; ++dim1Bin) {
      for (int dim2Bin = search_box[2]; dim2Bin < search_box[3] + 1; ++dim2Bin) {
        // get the id of this bin
        size_t binId = lt.getGlobalBinByBin(dim1Bin, dim2Bin);
        if constexpr (std::is_same_v<STRATEGY, HGCalScintillatorStrategy>)
          binId = lt.getGlobalBinByBin(dim1Bin, (dim2Bin % T::type::nRows));
        // get the size of this bin
        size_t binSize = lt[binId].size();

        // loop over all hits in this bin
        for (unsigned int j = 0; j < binSize; j++) {
          unsigned int otherId = lt[binId][j];
          float dist = distance(lt, i, otherId, layerId);
          bool foundHigher =
              (cellsOnLayer.rho[otherId] > cellsOnLayer.rho[i]) ||
              (cellsOnLayer.rho[otherId] == cellsOnLayer.rho[i] && cellsOnLayer.detid[otherId] > cellsOnLayer.detid[i]);
          if (foundHigher && dist <= i_delta) {
            // update i_delta
            i_delta = dist;
            // update i_nearestHigher
            i_nearestHigher = otherId;
          }
        }
      }
    }
    bool foundNearestHigherInSearchBox = (i_delta != maxDelta);
    if (foundNearestHigherInSearchBox) {
      cellsOnLayer.delta[i] = i_delta;
      cellsOnLayer.nearestHigher[i] = i_nearestHigher;
    } else {
      // otherwise delta is guaranteed to be larger outlierDeltaFactor_*delta_c
      // we can safely maximize delta to be maxDelta
      cellsOnLayer.delta[i] = maxDelta;
      cellsOnLayer.nearestHigher[i] = -1;
    }

    LogDebug("HGCalCLUEAlgo") << "Debugging calculateDistanceToHigher: \n"
                              << "  cell: " << i << " eta: " << cellsOnLayer.dim1[i] << " phi: " << cellsOnLayer.dim2[i]
                              << " energy: " << cellsOnLayer.weight[i] << " density: " << cellsOnLayer.rho[i]
                              << " nearest higher: " << cellsOnLayer.nearestHigher[i]
                              << " distance: " << cellsOnLayer.delta[i] << "\n";
  }
}

template <typename T, typename STRATEGY>
int HGCalCLUEAlgoT<T, STRATEGY>::findAndAssignClusters(const unsigned int layerId, float delta) {
  // this is called once per layer and endcap...
  // so when filling the cluster temporary vector of Hexels we resize each time
  // by the number  of clusters found. This is always equal to the number of
  // cluster centers...
  unsigned int nClustersOnLayer = 0;
  auto& cellsOnLayer = cells_[layerId];
  unsigned int numberOfCells = cellsOnLayer.detid.size();
  std::vector<int> localStack;
  // find cluster seeds and outlier
  for (unsigned int i = 0; i < numberOfCells; i++) {
    float rho_c = kappa_ * cellsOnLayer.sigmaNoise[i];
    // initialize clusterIndex
    cellsOnLayer.clusterIndex[i] = -1;
    bool isSeed = (cellsOnLayer.delta[i] > delta) && (cellsOnLayer.rho[i] >= rho_c);
    bool isOutlier = (cellsOnLayer.delta[i] > outlierDeltaFactor_ * delta) && (cellsOnLayer.rho[i] < rho_c);
    if (isSeed) {
      cellsOnLayer.clusterIndex[i] = nClustersOnLayer;
      cellsOnLayer.isSeed[i] = true;
      nClustersOnLayer++;
      localStack.push_back(i);

    } else if (!isOutlier) {
      cellsOnLayer.followers[cellsOnLayer.nearestHigher[i]].push_back(i);
    }
  }

  // need to pass clusterIndex to their followers
  while (!localStack.empty()) {
    int endStack = localStack.back();
    auto& thisSeed = cellsOnLayer.followers[endStack];
    localStack.pop_back();

    // loop over followers
    for (int j : thisSeed) {
      // pass id to a follower
      cellsOnLayer.clusterIndex[j] = cellsOnLayer.clusterIndex[endStack];
      // push this follower to localStack
      localStack.push_back(j);
    }
  }
  return nClustersOnLayer;
}

template <typename T, typename STRATEGY>
void HGCalCLUEAlgoT<T, STRATEGY>::computeThreshold() {
  // To support the TDR geometry and also the post-TDR one (v9 onwards), we
  // need to change the logic of the vectors containing signal to noise and
  // thresholds. The first 3 indices will keep on addressing the different
  // thicknesses of the Silicon detectors in CE_E , the next 3 indices will address
  // the thicknesses of the Silicon detectors in CE_H, while the last one, number 6 (the
  // seventh) will address the Scintillators. This change will support both
  // geometries at the same time.

  if (initialized_)
    return;  // only need to calculate thresholds once

  initialized_ = true;

  std::vector<double> dummy;

  dummy.resize(maxNumberOfThickIndices_ + !isNose_, 0);  // +1 to accomodate for the Scintillators
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
