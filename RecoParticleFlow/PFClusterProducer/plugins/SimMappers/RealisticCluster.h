#ifndef __RecoParticleFlow_PFClusterProducer_RealisticCluster_H__
#define __RecoParticleFlow_PFClusterProducer_RealisticCluster_H__

#include <array>
#include <vector>
#include <algorithm>

class RealisticCluster
{
        using Hit3DPosition = std::array<float,3>;

    public:

        // for each SimCluster and for each layer, we store the position of the most energetic hit of the simcluster in the layer

        struct LayerInfo
        {
            Hit3DPosition centerOfGravityAtLayer_;
            Hit3DPosition maxHitPosAtLayer_;
            float maxEnergyHitAtLayer_;
        };

        RealisticCluster():
            totalEnergy(0.f),
            exclusiveEnergy(0.f),
            visible(true)
        {

        }

        void increaseEnergy(float value)
        {
            totalEnergy+=value;
        }
        void increaseExclusiveEnergy(float value)
        {
            exclusiveEnergy+=value;
        }

        float getExclusiveEnergyFraction() const
        {
            float fraction = 0.f;
            if(totalEnergy>0.f){
                fraction = exclusiveEnergy/totalEnergy;
            }
            return fraction;
        }

        float getEnergy() const
        {
            return totalEnergy;
        }

        float getExclusiveEnergy() const
        {
            return exclusiveEnergy;
        }

        bool isVisible() const
        {
            return visible;
        }

        void setVisible(bool vis)
        {
            visible = vis;
        }

        void setCenterOfGravity(unsigned int layerId, const Hit3DPosition& position)
        {
            layerInfo_[layerId].centerOfGravityAtLayer_ = position;
        }

        Hit3DPosition getCenterOfGravity(unsigned int layerId) const
        {
            return layerInfo_[layerId].centerOfGravityAtLayer_ ;
        }

        bool setMaxEnergyHit(unsigned int layerId, float newEnergy, const Hit3DPosition position)
        {
            if (newEnergy > layerInfo_[layerId].maxEnergyHitAtLayer_)
            {
                layerInfo_[layerId].maxEnergyHitAtLayer_ = newEnergy;
                layerInfo_[layerId].maxHitPosAtLayer_ = position;
                return true;
            }
            else
                return false;
        }

        Hit3DPosition getMaxEnergyPosition (unsigned int layerId) const
        {
            return layerInfo_[layerId].maxHitPosAtLayer_;
        }

        float getMaxEnergy(unsigned int layerId) const
        {
            return layerInfo_[layerId].maxEnergyHitAtLayer_;
        }

        void setLayersNum(unsigned int numberOfLayers)
        {
            layerInfo_.resize(numberOfLayers);
        }

        unsigned int getLayersNum() const
        {
            return layerInfo_.size();
        }

        void addHitAndFraction(unsigned int hit, float fraction)
        {
            hitIdsAndFractions_.emplace_back(hit,fraction);
        }

        void modifyFractionForHitId(float fraction, unsigned int hitId)
        {
            auto it = std::find_if( hitIdsAndFractions_.begin(), hitIdsAndFractions_.end(),
                [&hitId](const std::pair<unsigned int, float>& element){ return element.first == hitId;} );

            it->second = fraction;
        }

        void modifyFractionByIndex(float fraction, unsigned int index)
        {
            hitIdsAndFractions_[index].second = fraction;
        }

        const std::vector< std::pair<unsigned int, float> > & hitsIdsAndFractions() const { return hitIdsAndFractions_; }

    private:
        std::vector<std::pair<unsigned int, float> > hitIdsAndFractions_;
        std::vector<LayerInfo> layerInfo_;

        float totalEnergy;
        float exclusiveEnergy;
        bool visible;
};

#endif
