#ifndef FASTSIM_STRAIGHTTRAJECTORY_H
#define FASTSIM_STRAIGHTTRAJECTORY_H

#include "FastSimulation/Propagation/interface/Trajectory.h"

namespace fastsim
{
    class StraightTrajectory : public Trajectory
    {
    public:
	StraightTrajectory(const Particle & particle) : Trajectory(particle) {;}
	bool crosses(const BarrelLayer & layer) const override {return true;}
	double nextCrossingTimeC(const BarrelLayer & layer) const override;
	void move(double deltaTimeC) override;
    };
}

#endif
