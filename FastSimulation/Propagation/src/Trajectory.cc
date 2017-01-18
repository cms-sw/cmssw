#include "DataFormats/Math/interface/LorentzVector.h"
#include "FastSimulation/Propagation/interface/StraightTrajectory.h"
#include "FastSimulation/Propagation/interface/HelixTrajectory.h"
#include "FastSimulation/Layer/interface/ForwardLayer.h"
#include "FastSimulation/Layer/interface/BarrelLayer.h"
#include "FastSimulation/NewParticle/interface/Particle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


const double fastsim::Trajectory::speedOfLight_ = 29.9792458; // [cm per ns]

fastsim::Trajectory::Trajectory(const fastsim::Particle & particle)
{
    position_ = particle.position();
    momentum_ = particle.momentum();
}

std::unique_ptr<fastsim::Trajectory> fastsim::Trajectory::createTrajectory(const fastsim::Particle & particle,double magneticFieldZ)
{
    if(particle.charge() == 0. || magneticFieldZ == 0.)
    {
	   LogDebug("FastSim") << "create straight trajectory";
	   return std::unique_ptr<fastsim::Trajectory>(new fastsim::StraightTrajectory(particle));
    }
    else if(std::abs(particle.momentum().Pt() / (speedOfLight_ * 1e-4 * particle.charge() * magneticFieldZ)) > 1e8){
       LogDebug("FastSim") << "create straight trajectory (huge radius)";
       return std::unique_ptr<fastsim::Trajectory>(new fastsim::StraightTrajectory(particle));
    }
    else
    {
	   LogDebug("FastSim") << "create helix trajectory";
	   return std::unique_ptr<fastsim::Trajectory>(new fastsim::HelixTrajectory(particle,magneticFieldZ));
    }
}


double fastsim::Trajectory::nextCrossingTimeC(const fastsim::Layer & layer) const
{
    if(layer.isForward())
    {
	return this->nextCrossingTimeC(static_cast<const fastsim::ForwardLayer &>(layer));
    }
    else
    {
	return this->nextCrossingTimeC(static_cast<const fastsim::BarrelLayer &>(layer));
    }
}


double fastsim::Trajectory::nextCrossingTimeC(const fastsim::ForwardLayer & layer) const
{
    if(layer.isOnSurface(position_))
    {
	   return -1;
    }

    // t = (z - z_0) / v_z
    // substitute: v_z = p_z / E * c  ( note: extra * c absorbed in p_z units)
    // => t*c = (z - z_0) / p_z * E
    double deltaTimeC = (layer.getZ() - position_.Z()) / momentum_.Z() * momentum_.E();
    return deltaTimeC > 0. ? deltaTimeC : -1.;
}


