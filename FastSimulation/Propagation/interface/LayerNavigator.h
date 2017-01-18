#ifndef FASTSIM_LAYERNAVIGATOR_H
#define FASTSIM_LAYERNAVIGATOR_H

#include "string"

namespace fastsim
{
    class Layer;
    class ForwardLayer;
    class BarrelLayer;
    class Geometry;
    class Particle;
    class LayerNavigator
    {
    public:
	LayerNavigator(const Geometry & geometry);
	// TODO: make the layer const
	bool moveParticleToNextLayer(Particle & particle,const Layer * & layer);
    private:
	const Geometry * const geometry_;
    const BarrelLayer * nextBarrelLayer_;
	const BarrelLayer * previousBarrelLayer_;
    const ForwardLayer * nextForwardLayer_;
	const ForwardLayer * previousForwardLayer_;
	static const std::string MESSAGECATEGORY;
    };
}

#endif
