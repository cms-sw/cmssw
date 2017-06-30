#ifndef __RecoParticleFlow_PFClusterProducer_RealisticHitToClusterAssociator_H__
#define __RecoParticleFlow_PFClusterProducer_RealisticHitToClusterAssociator_H__
/////////////////////////
// Author: Felice Pantaleo
// Date:   30/06/2017
// Email: felice@cern.ch
/////////////////////////
#include <vector>
#include "RealisticCluster.h"

namespace
{
//TODO: get number of layers per subdet from geometry
float getDecayLength(unsigned int layer)
{
    if (layer <= 28)
        return 2.f;
    if (layer > 28 && layer <= 40)
        return 1.5f;
    if (layer > 40)
        return 1.f;

    return 0.f;
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
            MCAssociatedSimCluster_.resize(numberOfHits);
            MCEnergyFraction_.resize(numberOfHits);
            HitToRealisticSimCluster_.resize(numberOfHits);
            HitToRealisticEnergyFraction_.resize(numberOfHits);
            distanceFromMaxHit_.resize(numberOfHits);
            maxHitPosAtLayer_.resize(numberOfSimClusters);
            maxEnergyHitAtLayer_.resize(numberOfSimClusters);
            for (unsigned int i = 0; i < numberOfSimClusters; ++i)
            {
                maxHitPosAtLayer_[i].resize(numberOfLayers);
                maxEnergyHitAtLayer_[i].resize(numberOfLayers, 0.f);

            }
            RealisticSimClusters_.resize(numberOfSimClusters);
        }

        void insertHitPosition(float x, float y, float z, unsigned int hitIndex)
        {
            hitPosition_[hitIndex] =
            {
                {   x,y,z}};

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
            MCAssociatedSimCluster_[hitIndex].push_back(scIdx);
            MCEnergyFraction_[hitIndex].push_back(fraction);
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

        void computeAssociation( float exclusiveFraction, bool useMCFractionsForExclEnergy)
        {
            //if more than exclusiveFraction of a hit's energy belongs to a cluster, that rechit is not counted as shared
            unsigned int numberOfHits = layerId_.size();
            std::vector<float> partialEnergies;

            for(unsigned int hitId = 0; hitId < numberOfHits; ++hitId)
            {
                partialEnergies.clear();
                unsigned int numberOfClusters = MCAssociatedSimCluster_[hitId].size();
                distanceFromMaxHit_[hitId].resize(numberOfClusters);

                unsigned int layer = layerId_[hitId];
                HitToRealisticSimCluster_[hitId].resize(numberOfClusters);
                HitToRealisticEnergyFraction_[hitId].resize(numberOfClusters);
                partialEnergies.resize(numberOfClusters,0.f);
                float energyDecayLength = getDecayLength(layer);
                float sumE = 0.f;
                if(numberOfClusters == 1)
                {
                    sumE = 1.f;
                    partialEnergies[0] = sumE;
                    distanceFromMaxHit_[hitId][0] = -XYdistanceFromMaxHit(hitId,MCAssociatedSimCluster_[hitId][0]);
                }
                else
                {
                    for(unsigned int clId = 0; clId < numberOfClusters; ++clId )
                    {

                        auto simClusterId = MCAssociatedSimCluster_[hitId][clId];
                        distanceFromMaxHit_[hitId][clId] = -XYdistanceFromMaxHit(hitId,simClusterId);
                        // partial energy is computed based on the distance from the maximum energy hit and its energy
                        // partial energy is only needed to compute a fraction and it's not the energy assigned to the cluster
                        if(maxEnergyHitAtLayer_[simClusterId][layer]>0.f)
                        partialEnergies[clId] = maxEnergyHitAtLayer_[simClusterId][layer] * std::exp(-distanceFromMaxHit_[hitId][clId]/energyDecayLength);

                        sumE += partialEnergies[clId];

                    }
                }
                if(sumE != 0.f)
                {
                    float invSumE = 1.f/sumE;
                    for(unsigned int clId = 0; clId < numberOfClusters; ++clId )
                    {

                        unsigned int simClusterIndex = MCAssociatedSimCluster_[hitId][clId];
                        HitToRealisticSimCluster_[hitId][clId] = simClusterIndex;
                        float assignedFraction = partialEnergies[clId]*invSumE;
                        HitToRealisticEnergyFraction_[hitId][clId] = assignedFraction;
                        float assignedEnergy = assignedFraction *totalEnergy_[hitId];
                        RealisticSimClusters_[simClusterIndex].increaseEnergy(assignedEnergy);
                        RealisticSimClusters_[simClusterIndex].addHitAndFraction(hitId, assignedFraction);
                        // if the hits energy belongs for more than exclusiveFraction to a cluster, also the cluster's
                        // exclusive energy is increased. The exclusive energy will be needed to evaluate if
                        // a realistic cluster will be invisible, i.e. absorbed by other clusters
                        bool isExclusive = ((!useMCFractionsForExclEnergy && (assignedFraction > exclusiveFraction) )
                                || (useMCFractionsForExclEnergy && (MCEnergyFraction_[hitId][clId] >exclusiveFraction )));
                        if(isExclusive)
                        {
                            RealisticSimClusters_[simClusterIndex].increaseExclusiveEnergy(assignedEnergy);
                        }
                    }
                }
            }
        }

        void findAndMergeInvisibleClusters(float invisibleFraction, float exclusiveFraction)
        {
            unsigned int numberOfRealSimClusters = RealisticSimClusters_.size();


            for(unsigned int clId= 0; clId < numberOfRealSimClusters; ++clId)
            {

                if(RealisticSimClusters_[clId].getExclusiveEnergyFraction() < invisibleFraction)
                {
                    RealisticSimClusters_[clId].setVisible(false);
                    auto& hAndF = RealisticSimClusters_[clId].hitsIdsAndFractions();
                    std::unordered_map < unsigned int, float> energyInNeighbors;
                    float totalSharedEnergy=0.f;
                    for(auto& elt : hAndF)
                    {
                        unsigned int hitId = elt.first;
                        float fraction = elt.second;
                        if(fraction<1.f)
                        {
                            float correction = 1.f/(1.f-fraction);
                            unsigned int numberOfClusters = HitToRealisticSimCluster_[hitId].size();

                            for(unsigned int i = 0; i< numberOfClusters; ++i)
                            {
                                if(HitToRealisticSimCluster_[hitId][i] != clId && RealisticSimClusters_[HitToRealisticSimCluster_[hitId][i]].isVisible())
                                {
                                    float oldEnergy = HitToRealisticEnergyFraction_[hitId][i]*totalEnergy_[hitId];
                                    HitToRealisticEnergyFraction_[hitId][i]*=correction;

                                    float newEnergy= HitToRealisticEnergyFraction_[hitId][i]*totalEnergy_[hitId];
                                    float sharedEnergy = newEnergy-oldEnergy;
                                    energyInNeighbors[HitToRealisticSimCluster_[hitId][i]] +=sharedEnergy;
                                    totalSharedEnergy +=sharedEnergy;
                                    RealisticSimClusters_[HitToRealisticSimCluster_[hitId][i]].increaseEnergy(sharedEnergy);
                                    RealisticSimClusters_[HitToRealisticSimCluster_[hitId][i]].modifyFractionForHitId(HitToRealisticEnergyFraction_[hitId][i], hitId);
                                }
                            }
                        }
                    }

                    for(auto& elt : hAndF)
                    {
                        unsigned int hitId = elt.first;
                        float fraction = elt.second;
                        if(fraction==1.f)
                        {
                            for (auto& pair: energyInNeighbors)
                            {
                                // hits that belonged completely to the absorbed cluster are redistributed
                                // based on the fraction of energy shared in the shared hits
                                float sharedFraction = pair.second/totalSharedEnergy;
                                float assignedEnergy = totalEnergy_[hitId]*sharedFraction;
                                RealisticSimClusters_[pair.first].increaseEnergy(assignedEnergy);
                                RealisticSimClusters_[pair.first].addHitAndFraction(hitId, sharedFraction);
                                if(sharedFraction > exclusiveFraction)
                                RealisticSimClusters_[pair.first].increaseExclusiveEnergy(assignedEnergy);

                            }

                        }
                    }

                }
            }
        }

        const std::vector< RealisticCluster > & realisticClusters() const
        {   return RealisticSimClusters_;}

        private:

        std::vector<Hit3DPosition> hitPosition_;
        std::vector<float> totalEnergy_;
        std::vector<unsigned int> layerId_;

        // MC association: for each hit, the indices of the SimClusters and their contributed
        // fraction to the energy of the hit is stored
        std::vector< std::vector<unsigned int> > MCAssociatedSimCluster_;
        std::vector< std::vector<float> > MCEnergyFraction_;
        // For each hit, the squared distance from the propagated simTrack to the layer is calculated for every SimCluster associated
        std::vector< std::vector<float> > distanceFromMaxHit_;

        // for each SimCluster and for each layer, we store the position of the most energetic hit of the simcluster in the layer
        std::vector< std::vector<Hit3DPosition> > maxHitPosAtLayer_;
        std::vector< std::vector<float> > maxEnergyHitAtLayer_;
        // Realistic association: for each hit, the indices of the RealisticClusters and their contributed
        // fraction to the energy of the hit is stored
        // There is one to one association between these realistic simclusters and simclusters
        std::vector< std::vector<unsigned int> > HitToRealisticSimCluster_;
        // for each hit, fractions of the energy associated to a realistic simcluster are computed
        std::vector< std::vector<float> > HitToRealisticEnergyFraction_;

        // the vector of the Realistic SimClusters
        std::vector< RealisticCluster > RealisticSimClusters_;

    };

#endif
