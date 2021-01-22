// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 11/2018
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/HGCalReco/interface/Common.h"
#include "PatternRecognitionbyCA.h"
#include "HGCDoublet.h"
#include "HGCGraph.h"
#include "DataFormats/Common/interface/ValueMap.h"

template <typename TILES>
void HGCGraphT<TILES>::makeAndConnectDoublets(const TILES &histo,
                                              const std::vector<TICLSeedingRegion> &regions,
                                              int nEtaBins,
                                              int nPhiBins,
                                              const std::vector<reco::CaloCluster> &layerClusters,
                                              const std::vector<float> &mask,
                                              const edm::ValueMap<std::pair<float, float>> &layerClustersTime,
                                              int deltaIEta,
                                              int deltaIPhi,
                                              float minCosTheta,
                                              float minCosPointing,
                                              float root_doublet_max_distance_from_seed_squared,
                                              float etaLimitIncreaseWindow,
                                              int skip_layers,
                                              int maxNumberOfLayers,
                                              float maxDeltaTime) {
  isOuterClusterOfDoublets_.clear();
  isOuterClusterOfDoublets_.resize(layerClusters.size());
  allDoublets_.clear();
  theRootDoublets_.clear();
  bool checkDistanceRootDoubletVsSeed = root_doublet_max_distance_from_seed_squared < 9999;
  float origin_eta;
  float origin_phi;
  for (const auto &r : regions) {
    bool isGlobal = (r.index == -1);
    auto zSide = r.zSide;
    int startEtaBin, endEtaBin, startPhiBin, endPhiBin;

    if (isGlobal) {
      startEtaBin = 0;
      startPhiBin = 0;
      endEtaBin = nEtaBins;
      endPhiBin = nPhiBins;
      origin_eta = 0;
      origin_phi = 0;
    } else {
      auto firstLayerOnZSide = maxNumberOfLayers * zSide;
      const auto &firstLayerHisto = histo[firstLayerOnZSide];
      origin_eta = r.origin.eta();
      origin_phi = r.origin.phi();
      int entryEtaBin = firstLayerHisto.etaBin(origin_eta);
      int entryPhiBin = firstLayerHisto.phiBin(origin_phi);
      // For track-seeded iterations, if the impact point is below a certain
      // eta-threshold, i.e., it has higher eta, make the initial search
      // window bigger in both eta and phi by one bin, to contain better low
      // energy showers.
      auto etaWindow = deltaIEta;
      auto phiWindow = deltaIPhi;
      if (std::abs(origin_eta) > etaLimitIncreaseWindow) {
        etaWindow++;
        phiWindow++;
        LogDebug("HGCGraph") << "Limit of Eta for increase: " << etaLimitIncreaseWindow
                             << " reached! Increasing inner search window" << std::endl;
      }
      startEtaBin = std::max(entryEtaBin - etaWindow, 0);
      endEtaBin = std::min(entryEtaBin + etaWindow + 1, nEtaBins);
      startPhiBin = entryPhiBin - phiWindow;
      endPhiBin = entryPhiBin + phiWindow + 1;
      if (verbosity_ > Guru) {
        LogDebug("HGCGraph") << " Entrance eta, phi: " << origin_eta << ", " << origin_phi
                             << " entryEtaBin: " << entryEtaBin << " entryPhiBin: " << entryPhiBin
                             << " globalBin: " << firstLayerHisto.globalBin(origin_eta, origin_phi)
                             << " on layer: " << firstLayerOnZSide << " startEtaBin: " << startEtaBin
                             << " endEtaBin: " << endEtaBin << " startPhiBin: " << startPhiBin
                             << " endPhiBin: " << endPhiBin << " phiBin(0): " << firstLayerHisto.phiBin(0.)
                             << " phiBin(" << M_PI / 2. << "): " << firstLayerHisto.phiBin(M_PI / 2.) << " phiBin("
                             << M_PI << "): " << firstLayerHisto.phiBin(M_PI) << " phiBin(" << -M_PI / 2.
                             << "): " << firstLayerHisto.phiBin(-M_PI / 2.) << " phiBin(" << -M_PI
                             << "): " << firstLayerHisto.phiBin(-M_PI) << " phiBin(" << 2. * M_PI
                             << "): " << firstLayerHisto.phiBin(2. * M_PI) << std::endl;
      }
    }

    for (int il = 0; il < maxNumberOfLayers - 1; ++il) {
      for (int outer_layer = 0; outer_layer < std::min(1 + skip_layers, maxNumberOfLayers - 1 - il); ++outer_layer) {
        int currentInnerLayerId = il + maxNumberOfLayers * zSide;
        int currentOuterLayerId = currentInnerLayerId + 1 + outer_layer;
        auto const &outerLayerHisto = histo[currentOuterLayerId];
        auto const &innerLayerHisto = histo[currentInnerLayerId];
        const int etaLimitIncreaseWindowBin = innerLayerHisto.etaBin(etaLimitIncreaseWindow);
        if (verbosity_ > Advanced) {
          LogDebug("HGCGraph") << "Limit of Eta for increase: " << etaLimitIncreaseWindow
                               << " at etaBin: " << etaLimitIncreaseWindowBin << std::endl;
        }

        for (int ieta = startEtaBin; ieta < endEtaBin; ++ieta) {
          auto offset = ieta * nPhiBins;
          for (int iphi_it = startPhiBin; iphi_it < endPhiBin; ++iphi_it) {
            int iphi = ((iphi_it % nPhiBins + nPhiBins) % nPhiBins);
            if (verbosity_ > Guru) {
              LogDebug("HGCGraph") << "Inner Global Bin: " << (offset + iphi)
                                   << " on layers I/O: " << currentInnerLayerId << "/" << currentOuterLayerId
                                   << " with clusters: " << innerLayerHisto[offset + iphi].size() << std::endl;
            }
            for (auto innerClusterId : innerLayerHisto[offset + iphi]) {
              // Skip masked clusters
              if (mask[innerClusterId] == 0.) {
                if (verbosity_ > Advanced)
                  LogDebug("HGCGraph") << "Skipping inner masked cluster " << innerClusterId << std::endl;
                continue;
              }

              // For global-seeded iterations, if the inner cluster is above a certain
              // eta-threshold, i.e., it has higher eta, make the outer search
              // window bigger in both eta and phi by one bin, to contain better low
              // energy showers. Track-Seeded iterations are excluded since, in
              // that case, the inner search window has already been enlarged.
              auto etaWindow = deltaIEta;
              auto phiWindow = deltaIPhi;
              if (isGlobal && ieta > etaLimitIncreaseWindowBin) {
                etaWindow++;
                phiWindow++;
                if (verbosity_ > Advanced) {
                  LogDebug("HGCGraph") << "Eta and Phi window increased by one" << std::endl;
                }
              }
              const auto etaRangeMin = std::max(0, ieta - etaWindow);
              const auto etaRangeMax = std::min(ieta + etaWindow + 1, nEtaBins);

              for (int oeta = etaRangeMin; oeta < etaRangeMax; ++oeta) {
                // wrap phi bin
                for (int phiRange = 0; phiRange < 2 * phiWindow + 1; ++phiRange) {
                  // The first wrapping is to take into account the
                  // cases in which we would have to seach in
                  // negative bins. The second wrap is mandatory to
                  // account for all other cases, since we add in
                  // between a full nPhiBins slot.
                  auto ophi = ((iphi + phiRange - phiWindow) % nPhiBins + nPhiBins) % nPhiBins;
                  if (verbosity_ > Guru) {
                    LogDebug("HGCGraph") << "Outer Global Bin: " << (oeta * nPhiBins + ophi)
                                         << " on layers I/O: " << currentInnerLayerId << "/" << currentOuterLayerId
                                         << " with clusters: " << innerLayerHisto[oeta * nPhiBins + ophi].size()
                                         << std::endl;
                  }
                  for (auto outerClusterId : outerLayerHisto[oeta * nPhiBins + ophi]) {
                    // Skip masked clusters
                    if (mask[outerClusterId] == 0.) {
                      if (verbosity_ > Advanced)
                        LogDebug("HGCGraph") << "Skipping outer masked cluster " << outerClusterId << std::endl;
                      continue;
                    }
                    auto doubletId = allDoublets_.size();
                    if (maxDeltaTime != -1 &&
                        !areTimeCompatible(innerClusterId, outerClusterId, layerClustersTime, maxDeltaTime)) {
                      if (verbosity_ > Advanced)
                        LogDebug("HGCGraph") << "Rejecting doublets due to timing!" << std::endl;
                      continue;
                    }
                    allDoublets_.emplace_back(innerClusterId, outerClusterId, doubletId, &layerClusters, r.index);
                    if (verbosity_ > Advanced) {
                      LogDebug("HGCGraph")
                          << "Creating doubletsId: " << doubletId << " layerLink in-out: [" << currentInnerLayerId
                          << ", " << currentOuterLayerId << "] clusterLink in-out: [" << innerClusterId << ", "
                          << outerClusterId << "]" << std::endl;
                    }
                    isOuterClusterOfDoublets_[outerClusterId].push_back(doubletId);
                    auto &neigDoublets = isOuterClusterOfDoublets_[innerClusterId];
                    auto &thisDoublet = allDoublets_[doubletId];
                    if (verbosity_ > Expert) {
                      LogDebug("HGCGraph")
                          << "Checking compatibility of doubletId: " << doubletId
                          << " with all possible inners doublets link by the innerClusterId: " << innerClusterId
                          << std::endl;
                    }
                    bool isRootDoublet = thisDoublet.checkCompatibilityAndTag(allDoublets_,
                                                                              neigDoublets,
                                                                              r.directionAtOrigin,
                                                                              minCosTheta,
                                                                              minCosPointing,
                                                                              verbosity_ > Advanced);
                    if (isRootDoublet and checkDistanceRootDoubletVsSeed) {
                      auto dEtaSquared = (layerClusters[innerClusterId].eta() - origin_eta);
                      dEtaSquared *= dEtaSquared;
                      auto dPhiSquared = (layerClusters[innerClusterId].phi() - origin_phi);
                      dPhiSquared *= dPhiSquared;
                      if (dEtaSquared + dPhiSquared > root_doublet_max_distance_from_seed_squared)
                        isRootDoublet = false;
                    }
                    if (isRootDoublet) {
                      theRootDoublets_.push_back(doubletId);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  // #ifdef FP_DEBUG
  if (verbosity_ > None) {
    LogDebug("HGCGraph") << "number of Root doublets " << theRootDoublets_.size() << " over a total number of doublets "
                         << allDoublets_.size() << std::endl;
  }
  // #endif
}

template <typename TILES>
bool HGCGraphT<TILES>::areTimeCompatible(int innerIdx,
                                         int outerIdx,
                                         const edm::ValueMap<std::pair<float, float>> &layerClustersTime,
                                         float maxDeltaTime) {
  float timeIn = layerClustersTime.get(innerIdx).first;
  float timeInE = layerClustersTime.get(innerIdx).second;
  float timeOut = layerClustersTime.get(outerIdx).first;
  float timeOutE = layerClustersTime.get(outerIdx).second;

  return (timeIn == -99. || timeOut == -99. ||
          std::abs(timeIn - timeOut) < maxDeltaTime * sqrt(timeInE * timeInE + timeOutE * timeOutE));
}

//also return a vector of seedIndex for the reconstructed tracksters
template <typename TILES>
void HGCGraphT<TILES>::findNtuplets(std::vector<HGCDoublet::HGCntuplet> &foundNtuplets,
                                    std::vector<int> &seedIndices,
                                    const unsigned int minClustersPerNtuplet,
                                    const bool outInDFS,
                                    unsigned int maxOutInHops) {
  HGCDoublet::HGCntuplet tmpNtuplet;
  tmpNtuplet.reserve(minClustersPerNtuplet);
  std::vector<std::pair<unsigned int, unsigned int>> outInToVisit;
  for (auto rootDoublet : theRootDoublets_) {
    tmpNtuplet.clear();
    outInToVisit.clear();
    int seedIndex = allDoublets_[rootDoublet].seedIndex();
    int outInHops = 0;
    allDoublets_[rootDoublet].findNtuplets(
        allDoublets_, tmpNtuplet, seedIndex, outInDFS, outInHops, maxOutInHops, outInToVisit);
    while (!outInToVisit.empty()) {
      allDoublets_[outInToVisit.back().first].findNtuplets(
          allDoublets_, tmpNtuplet, seedIndex, outInDFS, outInToVisit.back().second, maxOutInHops, outInToVisit);
      outInToVisit.pop_back();
    }

    if (tmpNtuplet.size() > minClustersPerNtuplet) {
      foundNtuplets.push_back(tmpNtuplet);
      seedIndices.push_back(seedIndex);
    }
  }
}

template class HGCGraphT<TICLLayerTiles>;
template class HGCGraphT<TICLLayerTilesHFNose>;
