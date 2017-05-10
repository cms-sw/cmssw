#ifndef FASTSIM_STRAIGHTTRAJECTORY_H
#define FASTSIM_STRAIGHTTRAJECTORY_H

#include "FastSimulation/SimplifiedGeometryPropagator/interface/Trajectory.h"

namespace fastsim
{
    class StraightTrajectory : public Trajectory
    {
    public:
	StraightTrajectory(const Particle & particle) : Trajectory(particle) {;}
	bool crosses(const BarrelSimplifiedGeometry & layer) const override {return true;}
	double nextCrossingTimeC(const BarrelSimplifiedGeometry & layer) const override;
	void move(double deltaTimeC) override;
    };
}

#endif
