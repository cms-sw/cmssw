#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalCLUEAlgo.h"

// Geometry
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
//
#include "DataFormats/CaloRecHit/interface/CaloID.h"
#include "tbb/task_arena.h"
#include "tbb/tbb.h"
#include <limits>

using namespace hgcal_clustering;

template <typename T>
void HGCalCLUEAlgoT<T>::getEventSetupPerAlgorithm(const edm::EventSetup& es) {
  cells_.clear();
  numberOfClustersPerLayer_.clear();
  cells_.resize(2 * (maxlayer_ + 1));
  numberOfClustersPerLayer_.resize(2 * (maxlayer_ + 1), 0);
}

template <typename T>
void HGCalCLUEAlgoT<T>::populate(const HGCRecHitCollection& hits) {
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
    cells_[layer].x.emplace_back(position.x());
    cells_[layer].y.emplace_back(position.y());
    if (!rhtools_.isOnlySilicon(layer)) {
      cells_[layer].isSi.emplace_back(rhtools_.isSilicon(detid));
      cells_[layer].eta.emplace_back(position.eta());
      cells_[layer].phi.emplace_back(position.phi());
    }  // else, isSilicon == true and eta phi values will not be used
    cells_[layer].weight.emplace_back(hgrh.energy());
    cells_[layer].sigmaNoise.emplace_back(sigmaNoise);
  }
}

template <typename T>
void HGCalCLUEAlgoT<T>::prepareDataStructures(unsigned int l) {
  auto cellsSize = cells_[l].detid.size();
  cells_[l].rho.resize(cellsSize, 0.f);
  cells_[l].delta.resize(cellsSize, 9999999);
  cells_[l].nearestHigher.resize(cellsSize, -1);
  cells_[l].clusterIndex.resize(cellsSize, -1);
  cells_[l].followers.resize(cellsSize);
  cells_[l].isSeed.resize(cellsSize, false);
  if (rhtools_.isOnlySilicon(l)) {
    cells_[l].isSi.resize(cellsSize, true);
    cells_[l].eta.resize(cellsSize, 0.f);
    cells_[l].phi.resize(cellsSize, 0.f);
  }
}

// Create a vector of Hexels associated to one cluster from a collection of
// HGCalRecHits - this can be used directly to make the final cluster list -
// this method can be invoked multiple times for the same event with different
// input (reset should be called between events)
template <typename T>
void HGCalCLUEAlgoT<T>::makeClusters() {
  // assign all hits in each layer to a cluster core
  tbb::this_task_arena::isolate([&] {
    tbb::parallel_for(size_t(0), size_t(2 * maxlayer_ + 2), [&](size_t i) {
      prepareDataStructures(i);
      T lt;
      lt.clear();
      lt.fill(cells_[i].x, cells_[i].y, cells_[i].eta, cells_[i].phi, cells_[i].isSi);
      float delta_c;  // maximum search distance (critical distance) for local
                      // density calculation
      if (i % maxlayer_ < lastLayerEE_)
        delta_c = vecDeltas_[0];
      else if (i % maxlayer_ < (firstLayerBH_ - 1))
        delta_c = vecDeltas_[1];
      else
        delta_c = vecDeltas_[2];
      float delta_r = vecDeltas_[3];
      LogDebug("HGCalCLUEAlgo") << "maxlayer: " << maxlayer_ << " lastLayerEE: " << lastLayerEE_
                                << " firstLayerBH: " << firstLayerBH_ << "\n";

      calculateLocalDensity(lt, i, delta_c, delta_r);
      calculateDistanceToHigher(lt, i, delta_c, delta_r);
      numberOfClustersPerLayer_[i] = findAndAssignClusters(i, delta_c, delta_r);
    });
  });
  //Now that we have the density per point we can store it
  for (unsigned int i = 0; i < 2 * maxlayer_ + 2; ++i) {
    setDensity(i);
  }
}

template <typename T>
std::vector<reco::BasicCluster> HGCalCLUEAlgoT<T>::getClusters(bool) {
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
      auto position = calculatePosition(cl, layerId);
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

template <typename T>
math::XYZPoint HGCalCLUEAlgoT<T>::calculatePosition(const std::vector<int>& v, const unsigned int layerId) const {
  float total_weight = 0.f;
  float x = 0.f;
  float y = 0.f;

  unsigned int maxEnergyIndex = 0;
  float maxEnergyValue = 0.f;

  auto& cellsOnLayer = cells_[layerId];

  // loop over hits in cluster candidate
  // determining the maximum energy hit
  for (auto i : v) {
    total_weight += cellsOnLayer.weight[i];
    if (cellsOnLayer.weight[i] > maxEnergyValue) {
      maxEnergyValue = cellsOnLayer.weight[i];
      maxEnergyIndex = i;
    }
  }

  // Si cell or Scintillator. Used to set approach and parameters
  auto thick = rhtools_.getSiThickIndex(cellsOnLayer.detid[maxEnergyIndex]);
  bool isSiliconCell = (thick != -1);

  // TODO: this is recomputing everything twice and overwriting the position with log weighting position
  if (isSiliconCell) {
    float total_weight_log = 0.f;
    float x_log = 0.f;
    float y_log = 0.f;
    for (auto i : v) {
      //for silicon only just use 1+6 cells = 1.3cm for all thicknesses
      if (distance2(i, maxEnergyIndex, layerId, false) > positionDeltaRho2_)
        continue;
      float rhEnergy = cellsOnLayer.weight[i];
      float Wi = std::max(thresholdW0_[thick] + std::log(rhEnergy / total_weight), 0.);
      x_log += cellsOnLayer.x[i] * Wi;
      y_log += cellsOnLayer.y[i] * Wi;
      total_weight_log += Wi;
    }

    total_weight = total_weight_log;
    x = x_log;
    y = y_log;
  } else {
    for (auto i : v) {
      float rhEnergy = cellsOnLayer.weight[i];

      x += cellsOnLayer.x[i] * rhEnergy;
      y += cellsOnLayer.y[i] * rhEnergy;
    }
  }
  if (total_weight != 0.) {
    float inv_tot_weight = 1.f / total_weight;
    return math::XYZPoint(
        x * inv_tot_weight, y * inv_tot_weight, rhtools_.getPosition(cellsOnLayer.detid[maxEnergyIndex]).z());
  } else
    return math::XYZPoint(0.f, 0.f, 0.f);
}

template <typename T>
void HGCalCLUEAlgoT<T>::calculateLocalDensity(const T& lt, const unsigned int layerId, float delta_c, float delta_r) {
  auto& cellsOnLayer = cells_[layerId];
  unsigned int numberOfCells = cellsOnLayer.detid.size();
  bool isOnlySi(false);
  if (rhtools_.isOnlySilicon(layerId))
    isOnlySi = true;

  for (unsigned int i = 0; i < numberOfCells; i++) {
    bool isSi = isOnlySi || cellsOnLayer.isSi[i];
    if (isSi) {
      float delta = delta_c;
      std::array<int, 4> search_box = lt.searchBox(
          cellsOnLayer.x[i] - delta, cellsOnLayer.x[i] + delta, cellsOnLayer.y[i] - delta, cellsOnLayer.y[i] + delta);

      for (int xBin = search_box[0]; xBin < search_box[1] + 1; ++xBin) {
        for (int yBin = search_box[2]; yBin < search_box[3] + 1; ++yBin) {
          int binId = lt.getGlobalBinByBin(xBin, yBin);
          size_t binSize = lt[binId].size();

          for (unsigned int j = 0; j < binSize; j++) {
            unsigned int otherId = lt[binId][j];
            bool otherSi = isOnlySi || cellsOnLayer.isSi[otherId];
            if (otherSi) {  //silicon cells cannot talk to scintillator cells
              if (distance(i, otherId, layerId, false) < delta) {
                cellsOnLayer.rho[i] += (i == otherId ? 1.f : 0.5f) * cellsOnLayer.weight[otherId];
              }
            }
          }
        }
      }
    } else {
      float delta = delta_r;
      std::array<int, 4> search_box = lt.searchBoxEtaPhi(cellsOnLayer.eta[i] - delta,
                                                         cellsOnLayer.eta[i] + delta,
                                                         cellsOnLayer.phi[i] - delta,
                                                         cellsOnLayer.phi[i] + delta);
      cellsOnLayer.rho[i] += cellsOnLayer.weight[i];
      float northeast(0), northwest(0), southeast(0), southwest(0), all(0);

      for (int etaBin = search_box[0]; etaBin < search_box[1] + 1; ++etaBin) {
        for (int phiBin = search_box[2]; phiBin < search_box[3] + 1; ++phiBin) {
          int binId = lt.getGlobalBinByBinEtaPhi(etaBin, phiBin);
          size_t binSize = lt[binId].size();

          for (unsigned int j = 0; j < binSize; j++) {
            unsigned int otherId = lt[binId][j];
            if (!cellsOnLayer.isSi[otherId]) {  //scintillator cells cannot talk to silicon cells
              if (distance(i, otherId, layerId, true) < delta) {
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
                                          << " otherIPhi: " << otherIPhi << " iPhi: " << iPhi
                                          << " otherIEta: " << otherIEta << " iEta: " << iEta << "\n";

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
      }
      float neighborsval = (std::max(northeast, northwest) > std::max(southeast, southwest))
                               ? std::max(northeast, northwest)
                               : std::max(southeast, southwest);
      if (use2x2_)
        cellsOnLayer.rho[i] += neighborsval;
      else
        cellsOnLayer.rho[i] += all;
    }
    LogDebug("HGCalCLUEAlgo") << "Debugging calculateLocalDensity: \n"
                              << "  cell: " << i << " isSilicon: " << cellsOnLayer.isSi[i]
                              << " eta: " << cellsOnLayer.eta[i] << " phi: " << cellsOnLayer.phi[i]
                              << " energy: " << cellsOnLayer.weight[i] << " density: " << cellsOnLayer.rho[i] << "\n";
  }
}

template <typename T>
void HGCalCLUEAlgoT<T>::calculateDistanceToHigher(const T& lt,
                                                  const unsigned int layerId,
                                                  float delta_c,
                                                  float delta_r) {
  auto& cellsOnLayer = cells_[layerId];
  unsigned int numberOfCells = cellsOnLayer.detid.size();
  bool isOnlySi(false);
  if (rhtools_.isOnlySilicon(layerId))
    isOnlySi = true;

  for (unsigned int i = 0; i < numberOfCells; i++) {
    bool isSi = isOnlySi || cellsOnLayer.isSi[i];
    // initialize delta and nearest higher for i
    float maxDelta = std::numeric_limits<float>::max();
    float i_delta = maxDelta;
    int i_nearestHigher = -1;
    if (isSi) {
      float delta = delta_c;
      // get search box for ith hit
      // guarantee to cover a range "outlierDeltaFactor_*delta_c"
      auto range = outlierDeltaFactor_ * delta;
      std::array<int, 4> search_box = lt.searchBox(
          cellsOnLayer.x[i] - range, cellsOnLayer.x[i] + range, cellsOnLayer.y[i] - range, cellsOnLayer.y[i] + range);
      // loop over all bins in the search box
      for (int xBin = search_box[0]; xBin < search_box[1] + 1; ++xBin) {
        for (int yBin = search_box[2]; yBin < search_box[3] + 1; ++yBin) {
          // get the id of this bin
          size_t binId = lt.getGlobalBinByBin(xBin, yBin);
          // get the size of this bin
          size_t binSize = lt[binId].size();

          // loop over all hits in this bin
          for (unsigned int j = 0; j < binSize; j++) {
            unsigned int otherId = lt[binId][j];
            bool otherSi = isOnlySi || cellsOnLayer.isSi[otherId];
            if (otherSi) {  //silicon cells cannot talk to scintillator cells
              float dist = distance(i, otherId, layerId, false);
              bool foundHigher = (cellsOnLayer.rho[otherId] > cellsOnLayer.rho[i]) ||
                                 (cellsOnLayer.rho[otherId] == cellsOnLayer.rho[i] &&
                                  cellsOnLayer.detid[otherId] > cellsOnLayer.detid[i]);
              // if dist == i_delta, then last comer being the nearest higher
              if (foundHigher && dist <= i_delta) {
                // update i_delta
                i_delta = dist;
                // update i_nearestHigher
                i_nearestHigher = otherId;
              }
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
    } else {
      //similar to silicon
      float delta = delta_r;
      auto range = outlierDeltaFactor_ * delta;
      std::array<int, 4> search_box = lt.searchBoxEtaPhi(cellsOnLayer.eta[i] - range,
                                                         cellsOnLayer.eta[i] + range,
                                                         cellsOnLayer.phi[i] - range,
                                                         cellsOnLayer.phi[i] + range);
      // loop over all bins in the search box
      for (int xBin = search_box[0]; xBin < search_box[1] + 1; ++xBin) {
        for (int yBin = search_box[2]; yBin < search_box[3] + 1; ++yBin) {
          // get the id of this bin
          size_t binId = lt.getGlobalBinByBinEtaPhi(xBin, yBin);
          // get the size of this bin
          size_t binSize = lt[binId].size();

          // loop over all hits in this bin
          for (unsigned int j = 0; j < binSize; j++) {
            unsigned int otherId = lt[binId][j];
            if (!cellsOnLayer.isSi[otherId]) {  //scintillator cells cannot talk to silicon cells
              float dist = distance(i, otherId, layerId, true);
              bool foundHigher = (cellsOnLayer.rho[otherId] > cellsOnLayer.rho[i]) ||
                                 (cellsOnLayer.rho[otherId] == cellsOnLayer.rho[i] &&
                                  cellsOnLayer.detid[otherId] > cellsOnLayer.detid[i]);
              // if dist == i_delta, then last comer being the nearest higher
              if (foundHigher && dist <= i_delta) {
                // update i_delta
                i_delta = dist;
                // update i_nearestHigher
                i_nearestHigher = otherId;
              }
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
    }
    LogDebug("HGCalCLUEAlgo") << "Debugging calculateDistanceToHigher: \n"
                              << "  cell: " << i << " isSilicon: " << cellsOnLayer.isSi[i]
                              << " eta: " << cellsOnLayer.eta[i] << " phi: " << cellsOnLayer.phi[i]
                              << " energy: " << cellsOnLayer.weight[i] << " density: " << cellsOnLayer.rho[i]
                              << " nearest higher: " << cellsOnLayer.nearestHigher[i]
                              << " distance: " << cellsOnLayer.delta[i] << "\n";
  }
}

template <typename T>
int HGCalCLUEAlgoT<T>::findAndAssignClusters(const unsigned int layerId, float delta_c, float delta_r) {
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
    bool isSi = rhtools_.isOnlySilicon(layerId) || cellsOnLayer.isSi[i];
    float delta = isSi ? delta_c : delta_r;

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

template <typename T>
void HGCalCLUEAlgoT<T>::computeThreshold() {
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

template <typename T>
void HGCalCLUEAlgoT<T>::setDensity(const unsigned int layerId) {
  auto& cellsOnLayer = cells_[layerId];
  unsigned int numberOfCells = cellsOnLayer.detid.size();
  for (unsigned int i = 0; i < numberOfCells; ++i)
    density_[cellsOnLayer.detid[i]] = cellsOnLayer.rho[i];
}

template <typename T>
Density HGCalCLUEAlgoT<T>::getDensity() {
  return density_;
}

// explicit template instantiation
template class HGCalCLUEAlgoT<HGCalLayerTiles>;
template class HGCalCLUEAlgoT<HFNoseLayerTiles>;
