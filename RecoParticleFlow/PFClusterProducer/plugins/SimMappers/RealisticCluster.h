#ifndef __RecoParticleFlow_PFClusterProducer_RealisticCluster_H__
#define __RecoParticleFlow_PFClusterProducer_RealisticCluster_H__

#include <array>
#include <vector>
#include <algorithm>

class RealisticCluster
{

    public:

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

        const std::vector< std::pair<unsigned int, float> > & hitsIdsAndFractions() const { return hitIdsAndFractions_; }



    private:
        std::vector<std::pair<unsigned int, float> > hitIdsAndFractions_;


        float totalEnergy;
        float exclusiveEnergy;
        bool visible;
};

#endif
