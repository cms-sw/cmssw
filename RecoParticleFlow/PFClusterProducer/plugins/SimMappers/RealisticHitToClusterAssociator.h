#ifndef __RecoParticleFlow_PFClusterProducer_RealisticHitToClusterAssociator_H__
#define __RecoParticleFlow_PFClusterProducer_RealisticHitToClusterAssociator_H__
/////////////////////////
// Author: Felice Pantaleo
// Date:   30/06/2017
// Email: felice@cern.ch
/////////////////////////
#include <vector>
#include <unordered_map>
#include "RealisticCluster.h"

namespace
{

float getDecayLength(unsigned int layer, unsigned int fhOffset, unsigned int bhOffset)
{
    constexpr float eeDecayLengthInLayer = 2.f;
    constexpr float fhDecayLengthInLayer = 1.5f;
    constexpr float bhDecayLengthInLayer = 1.f;

    if (layer <= fhOffset)
        return eeDecayLengthInLayer;
    else if (layer > fhOffset && layer <= bhOffset)
        return fhDecayLengthInLayer;
    else
        return bhDecayLengthInLayer;
}
}

class RealisticHitToClusterAssociator
{
        using Hit3DPosition = std::array<float,3>;
    public:
        void init(std::size_t numberOfHits, std::size_t numberOfSimClusters,
                std::size_t numberOfLayers)
        {
            hitPosition_.resize(numberOfHits);
            totalEnergy_.resize(numberOfHits);
            layerId_.resize(numberOfHits);
            mcAssociatedSimCluster_.resize(numberOfHits);
            mcEnergyFraction_.resize(numberOfHits);
            hitToRealisticSimCluster_.resize(numberOfHits);
            hitToRealisticEnergyFraction_.resize(numberOfHits);
            distanceFromMaxHit_.resize(numberOfHits);
            maxHitPosAtLayer_.resize(numberOfSimClusters);
            maxEnergyHitAtLayer_.resize(numberOfSimClusters);
            for (unsigned int i = 0; i < numberOfSimClusters; ++i)
            {
                maxHitPosAtLayer_[i].resize(numberOfLayers);
                maxEnergyHitAtLayer_[i].resize(numberOfLayers, 0.f);

            }
            realisticSimClusters_.resize(numberOfSimClusters);
        }

        void insertHitPosition(float x, float y, float z, unsigned int hitIndex)
        {
            hitPosition_[hitIndex] = {{x,y,z}};
        }

        void insertLayerId(unsigned int layerId, unsigned int hitIndex)
        {
            layerId_[hitIndex] = layerId;
        }

        void insertHitEnergy(float energy, unsigned int hitIndex)
        {
            totalEnergy_[hitIndex] = energy;
        }

        void insertSimClusterIdAndFraction(unsigned int scIdx, float fraction,
                unsigned int hitIndex, float associatedEnergy)
        {
            mcAssociatedSimCluster_[hitIndex].push_back(scIdx);
            mcEnergyFraction_[hitIndex].push_back(fraction);
            auto layerId = layerId_[hitIndex];
            if(associatedEnergy > maxEnergyHitAtLayer_[scIdx][layerId])
            {
                maxHitPosAtLayer_[scIdx][layerId] = hitPosition_[hitIndex];
                maxEnergyHitAtLayer_[scIdx][layerId] = associatedEnergy;
            }
        }

        float XYdistanceFromMaxHit(unsigned int hitId, unsigned int clusterId)
        {
            auto l = layerId_[hitId];
            auto& maxHitPosition = maxHitPosAtLayer_[clusterId][l];
            float distanceSquared = std::pow((hitPosition_[hitId][0] - maxHitPosition[0]),2) + std::pow((hitPosition_[hitId][1] - maxHitPosition[1]),2);
            return std::sqrt(distanceSquared);
        }

        void computeAssociation( float exclusiveFraction, bool useMCFractionsForExclEnergy, unsigned int fhOffset, unsigned int bhOffset)
        {
            //if more than exclusiveFraction of a hit's energy belongs to a cluster, that rechit is not counted as shared
            unsigned int numberOfHits = layerId_.size();
            std::vector<float> partialEnergies;

            for(unsigned int hitId = 0; hitId < numberOfHits; ++hitId)
            {
                partialEnergies.clear();
                std::vector<unsigned int> removeAssociation;

                unsigned int numberOfClusters = mcAssociatedSimCluster_[hitId].size();
                distanceFromMaxHit_[hitId].resize(numberOfClusters);

                hitToRealisticSimCluster_[hitId].resize(numberOfClusters);
                hitToRealisticEnergyFraction_[hitId].resize(numberOfClusters);
                if(numberOfClusters == 1)
                {

                    unsigned int simClusterIndex = mcAssociatedSimCluster_[hitId][0];
                    hitToRealisticSimCluster_[hitId][0] = simClusterIndex;
                    float assignedFraction = 1.f;
                    hitToRealisticEnergyFraction_[hitId][0] = assignedFraction;
                    float assignedEnergy = totalEnergy_[hitId];
                    realisticSimClusters_[simClusterIndex].increaseEnergy(assignedEnergy);
                    realisticSimClusters_[simClusterIndex].addHitAndFraction(hitId, assignedFraction);
                    realisticSimClusters_[simClusterIndex].increaseExclusiveEnergy(assignedEnergy);

                }
                else
                {
                    partialEnergies.resize(numberOfClusters,0.f);
                    unsigned int layer = layerId_[hitId];
                    float sumE = 0.f;
                    for(unsigned int clId = 0; clId < numberOfClusters; ++clId )
                    {
                        float energyDecayLength = getDecayLength(layer, fhOffset, bhOffset);
                        auto simClusterId = mcAssociatedSimCluster_[hitId][clId];
                        distanceFromMaxHit_[hitId][clId] = XYdistanceFromMaxHit(hitId,simClusterId);
                        // partial energy is computed based on the distance from the maximum energy hit and its energy
                        // partial energy is only needed to compute a fraction and it's not the energy assigned to the cluster
                        if(maxEnergyHitAtLayer_[simClusterId][layer]>0.f)
                        {
                            partialEnergies[clId] = maxEnergyHitAtLayer_[simClusterId][layer] * std::exp(-distanceFromMaxHit_[hitId][clId]/energyDecayLength);
                        }
                        sumE += partialEnergies[clId];
                    }
                    if(sumE > 0.f)
                    {
                        float invSumE = 1.f/sumE;
                        for(unsigned int clId = 0; clId < numberOfClusters; ++clId )
                        {
                            unsigned int simClusterIndex = mcAssociatedSimCluster_[hitId][clId];
                            hitToRealisticSimCluster_[hitId][clId] = simClusterIndex;
                            float assignedFraction = partialEnergies[clId]*invSumE;
                            if(assignedFraction > 1e-3)
                            {
                                hitToRealisticEnergyFraction_[hitId][clId] = assignedFraction;
                                float assignedEnergy = assignedFraction *totalEnergy_[hitId];
                                realisticSimClusters_[simClusterIndex].increaseEnergy(assignedEnergy);
                                realisticSimClusters_[simClusterIndex].addHitAndFraction(hitId, assignedFraction);
                                // if the hits energy belongs for more than exclusiveFraction to a cluster, also the cluster's
                                // exclusive energy is increased. The exclusive energy will be needed to evaluate if
                                // a realistic cluster will be invisible, i.e. absorbed by other clusters

                                if( (useMCFractionsForExclEnergy and mcEnergyFraction_[hitId][clId] > exclusiveFraction) or
                                        (!useMCFractionsForExclEnergy and assignedFraction > exclusiveFraction) )
                                {
                                    realisticSimClusters_[simClusterIndex].increaseExclusiveEnergy(assignedEnergy);
                                }
                            }
                            else
                            {
                                removeAssociation.push_back(simClusterIndex);
                            }
                        }
                    }
                }

                while(!removeAssociation.empty())
                {
                    auto clusterToRemove = removeAssociation.back();
                    removeAssociation.pop_back();
                    auto it = std::find(hitToRealisticSimCluster_[hitId].begin(), hitToRealisticSimCluster_[hitId].end(), clusterToRemove);
                    auto pos = it - hitToRealisticSimCluster_[hitId].begin();
                    hitToRealisticSimCluster_[hitId].erase(it);
                    hitToRealisticEnergyFraction_[hitId].erase(hitToRealisticEnergyFraction_[hitId].begin()+pos);
                }
            }
        }

        void findAndMergeInvisibleClusters(float invisibleFraction, float exclusiveFraction)
        {
            unsigned int numberOfRealSimClusters = realisticSimClusters_.size();

            for(unsigned int clId= 0; clId < numberOfRealSimClusters; ++clId)
            {

                if(realisticSimClusters_[clId].getExclusiveEnergyFraction() < invisibleFraction)
                {
                    realisticSimClusters_[clId].setVisible(false);
                    auto& hAndF = realisticSimClusters_[clId].hitsIdsAndFractions();
                    std::unordered_map < unsigned int, float> energyInNeighbors;
                    float totalSharedEnergy=0.f;

                    for(auto& elt : hAndF)
                    {
                        unsigned int hitId = elt.first;
                        float fraction = elt.second;
                        if(hitToRealisticSimCluster_[hitId].size() >1 && fraction < 1.f)
                        {
                            float correction = 1.f - fraction;
                            unsigned int numberOfClusters = hitToRealisticSimCluster_[hitId].size();
                            int clusterToRemove = -1;
                            for(unsigned int i = 0; i< numberOfClusters; ++i)
                            {
                                if(hitToRealisticSimCluster_[hitId][i] == clId)
                                {
                                    clusterToRemove = i;
                                }else
                                if(realisticSimClusters_[hitToRealisticSimCluster_[hitId][i]].isVisible())
                                {
                                    float oldFraction = hitToRealisticEnergyFraction_[hitId][i];
                                    float newFraction = oldFraction/correction;
                                    float oldEnergy = oldFraction*totalEnergy_[hitId];

                                    float newEnergy= newFraction*totalEnergy_[hitId];
                                    float sharedEnergy = newEnergy-oldEnergy;
                                    energyInNeighbors[hitToRealisticSimCluster_[hitId][i]] +=sharedEnergy;
                                    totalSharedEnergy +=sharedEnergy;
                                    realisticSimClusters_[hitToRealisticSimCluster_[hitId][i]].increaseEnergy(sharedEnergy);
                                    realisticSimClusters_[hitToRealisticSimCluster_[hitId][i]].modifyFractionForHitId(newFraction, hitId);
                                    hitToRealisticEnergyFraction_[hitId][i] = newFraction;
                                    if(newFraction > exclusiveFraction)
                                    {
                                        realisticSimClusters_[hitToRealisticSimCluster_[hitId][i]].increaseExclusiveEnergy(sharedEnergy);
                                        if(oldFraction <=exclusiveFraction)
                                        {
                                            realisticSimClusters_[hitToRealisticSimCluster_[hitId][i]].increaseExclusiveEnergy(oldEnergy);
                                        }
                                    }

                                }
                            }
                            hitToRealisticEnergyFraction_[hitId][clusterToRemove] = 0.f;
                            realisticSimClusters_[hitToRealisticSimCluster_[hitId][clusterToRemove]].modifyFractionForHitId(0.f, hitId);
                        }
                    }

                    for(auto& elt : hAndF)
                    {
                        unsigned int hitId = elt.first;
                        if(hitToRealisticSimCluster_[hitId].size()==1 and totalSharedEnergy > 0.f)
                        {
                            for (auto& pair: energyInNeighbors)
                            {
                                // hits that belonged completely to the absorbed cluster are redistributed
                                // based on the fraction of energy shared in the shared hits
                                float sharedFraction = pair.second/totalSharedEnergy;
                                float assignedEnergy = totalEnergy_[hitId]*sharedFraction;
                                realisticSimClusters_[pair.first].increaseEnergy(assignedEnergy);
                                realisticSimClusters_[pair.first].addHitAndFraction(hitId, sharedFraction);
                                hitToRealisticSimCluster_[hitId].push_back(pair.first);
                                hitToRealisticEnergyFraction_[hitId].push_back(sharedFraction);
                                hitToRealisticEnergyFraction_[hitId][0] = 0.f;
                                if(sharedFraction > exclusiveFraction)
                                    realisticSimClusters_[pair.first].increaseExclusiveEnergy(assignedEnergy);
                            }

                        }
                    }

                }
            }
        }

        const std::vector< RealisticCluster > & realisticClusters() const
        {   return realisticSimClusters_;}

    private:

        std::vector<Hit3DPosition> hitPosition_;
        std::vector<float> totalEnergy_;
        std::vector<unsigned int> layerId_;

        // MC association: for each hit, the indices of the SimClusters and their contributed
        // fraction to the energy of the hit is stored
        std::vector< std::vector<unsigned int> > mcAssociatedSimCluster_;
        std::vector< std::vector<float> > mcEnergyFraction_;
        // For each hit, the squared distance from the propagated simTrack to the layer is calculated for every SimCluster associated
        std::vector< std::vector<float> > distanceFromMaxHit_;

        // for each SimCluster and for each layer, we store the position of the most energetic hit of the simcluster in the layer
        std::vector< std::vector<Hit3DPosition> > maxHitPosAtLayer_;
        std::vector< std::vector<float> > maxEnergyHitAtLayer_;
        // Realistic association: for each hit, the indices of the RealisticClusters and their contributed
        // fraction to the energy of the hit is stored
        // There is one to one association between these realistic simclusters and simclusters
        std::vector< std::vector<unsigned int> > hitToRealisticSimCluster_;
        // for each hit, fractions of the energy associated to a realistic simcluster are computed
        std::vector< std::vector<float> > hitToRealisticEnergyFraction_;

        // the vector of the Realistic SimClusters
        std::vector< RealisticCluster > realisticSimClusters_;

    };

#endif
