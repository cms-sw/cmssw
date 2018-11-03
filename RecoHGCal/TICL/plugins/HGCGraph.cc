// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 11/2018
// Copyright CERN
#include "RecoHGCal/TICL/interface/TICLConstants.h"

#include "HGCGraph.h"
#include "HGCDoublet.h"

void HGCGraph::makeAndConnectDoublets(const std::vector<std::vector<std::vector<unsigned int>>> &h, int nEtaBins,
                                      int nPhiBins, const std::vector<reco::CaloCluster> &layerClusters, int nClusters, int deltaIEta, int deltaIPhi, float minCosTheta)
{
    isOuterClusterOfDoublets_.clear();
    isOuterClusterOfDoublets_.resize(nClusters);
    allDoublets_.clear();
    theRootDoublets_.clear();
    for (int zSide = 0; zSide < 2; ++zSide)
    {
        for (int il = 0; il < ticlConstants::maxNumberOfLayers - 1; ++il)
        {
            int currentInnerLayerId = il + ticlConstants::maxNumberOfLayers * zSide;
            int currentOuterLayerId = currentInnerLayerId + 1;
            auto &outerLayerHisto = h[currentOuterLayerId];
            auto &innerLayerHisto = h[currentInnerLayerId];

            for (int oeta = 0; oeta < nEtaBins; ++oeta)
            {
                auto offset = oeta * nPhiBins;
                for (int ophi = 0; ophi < nPhiBins; ++ophi)
                {
                    for (auto outerClusterId : outerLayerHisto[offset + ophi])
                    {
                        const auto etaRangeMin = std::max(0, oeta - deltaIEta);
                        const auto etaRangeMax = std::min(oeta + deltaIEta, nEtaBins);

                        for (int ieta = etaRangeMin; ieta < etaRangeMax - etaRangeMin; ++ieta)
                        {

                            //wrap phi bin
                            for (int phiRange = 0; phiRange < 2 * deltaIPhi + 1; ++phiRange)
                            {
                                auto iphi = ((ophi + phiRange - deltaIPhi) % nPhiBins + nPhiBins) % nPhiBins;
                                for (auto innerClusterId : innerLayerHisto[ieta * nPhiBins + iphi])
                                {
                                    auto doubletId = allDoublets_.size();
                                    allDoublets_.emplace_back(innerClusterId, outerClusterId, doubletId, &layerClusters);
                                    isOuterClusterOfDoublets_[outerClusterId].push_back(doubletId);
                                    auto &neigDoublets = isOuterClusterOfDoublets_[innerClusterId];
                                    auto &thisDoublet = allDoublets_[doubletId];
                                    bool isRootDoublet = thisDoublet.checkCompatibilityAndTag(allDoublets_, neigDoublets, minCosTheta);
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
    #ifdef FP_DEBUG
    std::cout << "number of Root doublets " << theRootDoublets_.size() << " over a total number of doublets " << allDoublets_.size() << std::endl;
    #endif
}


void HGCGraph::findNtuplets(std::vector<HGCDoublet::HGCntuplet> &foundNtuplets, const unsigned int minClustersPerNtuplet)
{
    HGCDoublet::HGCntuplet tmpNtuplet;
    tmpNtuplet.reserve(minClustersPerNtuplet);
    for (auto rootDoublet : theRootDoublets_)
    {
        tmpNtuplet.clear();
        allDoublets_[rootDoublet].findNtuplets(allDoublets_, tmpNtuplet);
        if (tmpNtuplet.size() > minClustersPerNtuplet)
        {
            foundNtuplets.push_back(tmpNtuplet);

        }
    }
}


