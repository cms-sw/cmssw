#ifndef FASTSIM_TRAJECTORY_H
#define FASTSIM_TRAJECTORY_H

#include "memory"

#include "DataFormats/Math/interface/LorentzVector.h"

namespace fastsim
{
    class Layer;
    class BarrelLayer;
    class ForwardLayer;
    class Particle;
    class Trajectory
    {
    public:
	static std::unique_ptr<Trajectory> createTrajectory(const fastsim::Particle & particle,const double magneticFieldZ);
	virtual bool crosses(const BarrelLayer & layer) const = 0;
	const math::XYZTLorentzVector & getPosition(){return position_;}
	const math::XYZTLorentzVector & getMomentum(){return momentum_;}
	double nextCrossingTimeC(const Layer & layer) const;
	double nextCrossingTimeC(const ForwardLayer & layer) const;
	virtual double nextCrossingTimeC(const BarrelLayer & layer) const = 0;
	virtual void move(double deltaTC) = 0;
    protected:
	Trajectory(const fastsim::Particle & particle);
	math::XYZTLorentzVector position_;
	math::XYZTLorentzVector momentum_;
	static const double speedOfLight_; // in cm / ns
	static const double epsiloneTimeC_;
    };
}

#endif
