// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 11/2018
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/HGCalReco/interface/Common.h"
#include "PatternRecognitionbyCA.h"
#include "HGCDoublet.h"
#include "HGCGraph.h"
#include "DataFormats/Common/interface/ValueMap.h"

void HGCGraph::makeAndConnectDoublets(const TICLLayerTiles &histo,
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
                                      int missing_layers,
                                      int maxNumberOfLayers,
                                      float maxDeltaTime) {
  isOuterClusterOfDoublets_.clear();
  isOuterClusterOfDoublets_.resize(layerClusters.size());
  allDoublets_.clear();
  theRootDoublets_.clear();
  for (const auto &r : regions) {
    bool isGlobal = (r.index == -1);
    auto zSide = r.zSide;
    int startEtaBin, endEtaBin, startPhiBin, endPhiBin;

    if (isGlobal) {
      startEtaBin = 0;
      startPhiBin = 0;
      endEtaBin = nEtaBins;
      endPhiBin = nPhiBins;
    } else {
      auto firstLayerOnZSide = maxNumberOfLayers * zSide;
      const auto &firstLayerHisto = histo[firstLayerOnZSide];

      int entryEtaBin = firstLayerHisto.etaBin(r.origin.eta());
      int entryPhiBin = firstLayerHisto.phiBin(r.origin.phi());
      startEtaBin = std::max(entryEtaBin - deltaIEta, 0);
      endEtaBin = std::min(entryEtaBin + deltaIEta + 1, nEtaBins);
      startPhiBin = entryPhiBin - deltaIPhi;
      endPhiBin = entryPhiBin + deltaIPhi + 1;
    }

    for (int il = 0; il < maxNumberOfLayers - 1; ++il) {
      for (int outer_layer = 0; outer_layer < std::min(1 + missing_layers, maxNumberOfLayers - 1 - il); ++outer_layer) {
        int currentInnerLayerId = il + maxNumberOfLayers * zSide;
        int currentOuterLayerId = currentInnerLayerId + 1 + outer_layer;
        auto const &outerLayerHisto = histo[currentOuterLayerId];
        auto const &innerLayerHisto = histo[currentInnerLayerId];

        for (int ieta = startEtaBin; ieta < endEtaBin; ++ieta) {
          auto offset = ieta * nPhiBins;
          for (int iphi_it = startPhiBin; iphi_it < endPhiBin; ++iphi_it) {
            int iphi = ((iphi_it % nPhiBins + nPhiBins) % nPhiBins);
            for (auto innerClusterId : innerLayerHisto[offset + iphi]) {
              // Skip masked clusters
              if (mask[innerClusterId] == 0.)
                continue;
              const auto etaRangeMin = std::max(0, ieta - deltaIEta);
              const auto etaRangeMax = std::min(ieta + deltaIEta + 1, nEtaBins);

              for (int oeta = etaRangeMin; oeta < etaRangeMax; ++oeta) {
                // wrap phi bin
                for (int phiRange = 0; phiRange < 2 * deltaIPhi + 1; ++phiRange) {
                  // The first wrapping is to take into account the
                  // cases in which we would have to seach in
                  // negative bins. The second wrap is mandatory to
                  // account for all other cases, since we add in
                  // between a full nPhiBins slot.
                  auto ophi = ((iphi + phiRange - deltaIPhi) % nPhiBins + nPhiBins) % nPhiBins;
                  for (auto outerClusterId : outerLayerHisto[oeta * nPhiBins + ophi]) {
                    // Skip masked clusters
                    if (mask[outerClusterId] == 0.)
                      continue;
                    auto doubletId = allDoublets_.size();
                    if (maxDeltaTime != -1 &&
                        !areTimeCompatible(innerClusterId, outerClusterId, layerClustersTime, maxDeltaTime))
                      continue;
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
                    if (isRootDoublet)
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
  // #ifdef FP_DEBUG
  if (verbosity_ > None) {
    LogDebug("HGCGraph") << "number of Root doublets " << theRootDoublets_.size() << " over a total number of doublets "
                         << allDoublets_.size() << std::endl;
  }
  // #endif
}

bool HGCGraph::areTimeCompatible(int innerIdx,
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
void HGCGraph::findNtuplets(std::vector<HGCDoublet::HGCntuplet> &foundNtuplets,
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
