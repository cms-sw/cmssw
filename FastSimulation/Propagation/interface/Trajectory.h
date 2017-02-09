#ifndef FASTSIM_TRAJECTORY_H
#define FASTSIM_TRAJECTORY_H

#include "memory"

#include "DataFormats/Math/interface/LorentzVector.h"

namespace fastsim
{
    class SimplifiedGeometry;
    class BarrelSimplifiedGeometry;
    class ForwardSimplifiedGeometry;
    class Particle;
    class Trajectory
    {
    public:
	static std::unique_ptr<Trajectory> createTrajectory(const fastsim::Particle & particle,const double magneticFieldZ);
	virtual bool crosses(const BarrelSimplifiedGeometry & layer) const = 0;
	const math::XYZTLorentzVector & getPosition(){return position_;}
	const math::XYZTLorentzVector & getMomentum(){return momentum_;}
	double nextCrossingTimeC(const SimplifiedGeometry & layer) const;
	double nextCrossingTimeC(const ForwardSimplifiedGeometry & layer) const;
	virtual double nextCrossingTimeC(const BarrelSimplifiedGeometry & layer) const = 0;
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
