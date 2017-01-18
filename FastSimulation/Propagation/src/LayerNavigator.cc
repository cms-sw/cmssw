#include "FastSimulation/Propagation/interface/LayerNavigator.h"

#include "vector"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FastSimulation/Geometry/interface/Geometry.h"
#include "FastSimulation/Layer/interface/BarrelLayer.h"
#include "FastSimulation/Layer/interface/ForwardLayer.h"
#include "FastSimulation/Propagation/interface/LayerNavigator.h"
#include "FastSimulation/Propagation/interface/Trajectory.h"      // new class, to be defined, based on ParticlePropagator
#include "FastSimulation/NewParticle/interface/Particle.h"

/**
// find the next layer that the particle will cross
//
// motivation for a new algorithm
//
//    - old algorithm is flawed
//    - the new algorithm allows to put material and instruments on any plane perpendicular to z, or on any cylinder with the z-axis as axis 
//    - while the old algorith, with the requirement of nested layers, forbids the introduction of long narrow cylinders, required for a decent simulation of material in front of HF
// 
// definitions:
//
//    the geometry is described by 2 sets of layers:
//    - forward layers: 
//          flat layers, perpendicular to the z-axis, positioned at a given z
//          these layers have material / instruments between a given materialMinR and materialMaxR
//          no 2 forward layers should have the same z-position
//    - barrel layers: 
//          cylindrically shaped layers, with the z-axis as axis, infinitely long
//          these layers have material / instruments for |z| < materialMaxAbsZ
//          no 2 barrel layers should have the same radius
//    - forward(barrel) layers are ordered according to increasing z (r)

// principle
//    - neutral particles follow a straight trajectory
//    - charged particles follow a helix-shaped trajectory:
//          constant speed along the z-axis
//          circular path in the x-y plane
//    => the next layer that the particle will cross is among the following 3 layers
//    - closest forward layer with
//         - z >(<) particle.z() for particles moving in the positive(negative) direction
//    - closest barrel layer with r < particle.r
//    - closest barrel layer with r > particle.r  

// algorithm
//    - find the 3 candidate layers 
//    - find the earliest positive intersection time for each of the 3 candidate layers
//    - move the particle to the earliest intersection time
//    - select and return the layer with the earliest positive intersection time
//
// notes
//    - the implementation of the algorithm can probably be optimised, e.g.
//       - one can probably gain time in moveToNextLayer if LayerNavigator is aware of the candidate layers of the previous call to moveToNextLayer
//       - for straight tracks, the optimal strategy to find the next layer might be very different
**/

const std::string fastsim::LayerNavigator::MESSAGECATEGORY = "FastSimulation";

fastsim::LayerNavigator::LayerNavigator(const fastsim::Geometry & geometry)
    : geometry_(&geometry)
    , nextBarrelLayer_(0)
    , previousBarrelLayer_(0)
    , nextForwardLayer_(0)
    , previousForwardLayer_(0)
{;}

bool fastsim::LayerNavigator::moveParticleToNextLayer(fastsim::Particle & particle,const fastsim::Layer * & layer)
{
    LogDebug(MESSAGECATEGORY) << "   moveToNextLayer called";

    // if the layer is provided, the particle must be on it
    if(layer)
    {	
		if(!layer->isOnSurface(particle.position()))
		{
		    throw cms::Exception("FastSimulation") << "If layer is provided, particle must be on layer."
		    << "\n   Layer: " << *layer
		    << "\n   Particle: " << particle;
		}
    }

    // magnetic field at the current position of the particle
    double magneticFieldZ = layer ? layer->getMagneticFieldZ(particle.position()) : geometry_->getMagneticFieldZ(particle.position());
    LogDebug(MESSAGECATEGORY) << "   magnetic field z component:" << magneticFieldZ;

    // particle moves inwards?
    bool particleMovesInwards = particle.momentum().X()*particle.position().X() + particle.momentum().Y()*particle.position().Y() < 0;
    
    //
    //  update nextBarrelLayer and nextForwardLayer
    //

    // first time
    if(!layer)
    {		
		LogDebug(MESSAGECATEGORY) << "      called for first time";

		//
		// find the narrowest barrel layers with
		// layer.r > particle.r (the closest layer with layer.r < particle.r will then be considered, too)
		// assume barrel layers are ordered with increasing r
		//
		for(const auto & layer : geometry_->barrelLayers())
		{
			if(layer->isOnSurface(particle.position())){
				if(particleMovesInwards){
					nextBarrelLayer_ = layer.get();
					break;
				}else{
					continue;
				}
			}

		    if(particle.position().Pt() < layer->getRadius())
		    {
				nextBarrelLayer_ = layer.get();
				break;
		    }

			previousBarrelLayer_ = layer.get();
		}

		// 
		//  find the forward layer with smallest z with
		//  layer.z > particle z (the closest layer with layer.z < particle.z will then be considered, too)
		//
		for(const auto & layer : geometry_->forwardLayers())
		{
			if(layer->isOnSurface(particle.position())){
				if(particle.momentum().Z() < 0){
					nextForwardLayer_ = layer.get();
					break;
				}else{
					continue;
				}
			}

		    if(particle.position().Z() < layer->getZ())
		    {
				nextForwardLayer_ = layer.get();
				break;
		    }

			previousForwardLayer_ = layer.get();
		}
    }
    //
    // last move worked, let's update
    //
    else
    {
		LogDebug(MESSAGECATEGORY) << "      ordinary call";
		// barrel layer was hit
		if(layer == nextBarrelLayer_)
		{
		    if(!particleMovesInwards)
		    {
		    	previousBarrelLayer_ = nextBarrelLayer_;
				nextBarrelLayer_ = geometry_->nextLayer(nextBarrelLayer_);
		    }
		}
		else if(layer == previousBarrelLayer_)
		{
		    if(particleMovesInwards)
		    {
				nextBarrelLayer_ = previousBarrelLayer_;
				previousBarrelLayer_ = geometry_->previousLayer(previousBarrelLayer_);
		    }
		}
		// forward layer was hit
		else if(layer == nextForwardLayer_)
		{
		    if(particle.momentum().Z() > 0)
		    {
				previousForwardLayer_ = nextForwardLayer_;
				nextForwardLayer_ = geometry_->nextLayer(nextForwardLayer_);
		    }
		}
		else if(layer == previousForwardLayer_)
		{
		    if(particle.momentum().Z() < 0)
		    {
				nextForwardLayer_ = previousForwardLayer_;
				previousForwardLayer_ = geometry_->previousLayer(previousForwardLayer_);
		    }
		}
		layer = 0;
    }

    //
    // move particle to first hit with one of the enclosing layers
    //
    
    // TODO: for straight tracks you KNOW in advance wether next or previous barrel layer will be hit: use that information!

    
    LogDebug(MESSAGECATEGORY) << "   particle between BarrelLayers: " << (previousBarrelLayer_ ? previousBarrelLayer_->index() : -1) << "/" << (nextBarrelLayer_ ? nextBarrelLayer_->index() : -1) << " (total: "<< geometry_->barrelLayers().size() <<")"
			      << "\n   particle between ForwardLayers: " << (previousForwardLayer_ ? previousForwardLayer_->index() : -1) << "/" << (nextForwardLayer_ ? nextForwardLayer_->index() : -1) << " (total: "<< geometry_->forwardLayers().size() <<")";
    
    // calculate and store some variables related to the particle's trajectory
    std::unique_ptr<fastsim::Trajectory> trajectory = Trajectory::createTrajectory(particle,magneticFieldZ);
    
    // now let's try to move the particle to one of the enclosing layers
    std::vector<const fastsim::Layer*> layers;
    if(nextBarrelLayer_) 
    {
		layers.push_back(nextBarrelLayer_);
    }
    if(previousBarrelLayer_)
    {
		layers.push_back(previousBarrelLayer_);
    }
    if(particle.momentum().Z() > 0)
    {
		if(nextForwardLayer_)
		{
		    layers.push_back(nextForwardLayer_);
		}
    }
    else
    {
		if(previousForwardLayer_)
		{
		    layers.push_back(previousForwardLayer_);
		}
    }
    
    double deltaTime = -1;
    for(auto _layer : layers)
    {
		double tempDeltaTime = trajectory->nextCrossingTimeC(*_layer);
		LogDebug(MESSAGECATEGORY) << "   particle crosses layer " << *_layer << " at time " << tempDeltaTime;
		if(tempDeltaTime > 0 && (layer == 0 || tempDeltaTime<deltaTime || deltaTime < 0))
		{
		    layer = _layer;
		    deltaTime = tempDeltaTime;
		}
    }

    // TODO : review time unit: ct or just t?
    double properDeltaTime = deltaTime / particle.gamma();
    if(!particle.isStable() && properDeltaTime > particle.remainingProperLifeTime())
    {
		deltaTime = particle.remainingProperLifeTime() * particle.gamma();
		particle.setRemainingProperLifeTime(0.);
    }

    // temporary, to get rid of additional hits since there is no ecal and stuff yet
    if(deltaTime > 100) return 0;

    // move particle in space, time and momentum
    if(layer)
    {
		trajectory->move(deltaTime);
		particle.position() = trajectory->getPosition();
		particle.momentum() = trajectory->getMomentum();
		LogDebug(MESSAGECATEGORY) << "    moved particle to layer: " << *layer;
    }

    // return true / false if propagations succeeded /failed
    LogDebug(MESSAGECATEGORY) << "    success: " << bool(layer);
    return layer;
}

	
