#ifndef FASTSIM_LAYERNAVIGATOR_H
#define FASTSIM_LAYERNAVIGATOR_H

#include "string"

namespace fastsim
{
    class SimplifiedGeometry;
    class ForwardSimplifiedGeometry;
    class BarrelSimplifiedGeometry;
    class Geometry;
    class Particle;
    class LayerNavigator
    {
    public:
	LayerNavigator(const Geometry & geometry);
	bool moveParticleToNextLayer(Particle & particle,const SimplifiedGeometry * & layer);
    private:
	const Geometry * const geometry_;
    const BarrelSimplifiedGeometry * nextBarrelLayer_;
	const BarrelSimplifiedGeometry * previousBarrelLayer_;
    const ForwardSimplifiedGeometry * nextForwardLayer_;
	const ForwardSimplifiedGeometry * previousForwardLayer_;
	static const std::string MESSAGECATEGORY;
    };
}

#endif
