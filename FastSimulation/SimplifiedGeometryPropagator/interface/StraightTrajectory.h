#ifndef FASTSIM_STRAIGHTTRAJECTORY_H
#define FASTSIM_STRAIGHTTRAJECTORY_H

#include "FastSimulation/SimplifiedGeometryPropagator/interface/Trajectory.h"


///////////////////////////////////////////////
// Author: L. Vanelderen, S. Kurz
// Date: 29 May 2017
//////////////////////////////////////////////////////////


namespace fastsim
{
    //! Mathematical representation of a straight trajectory.
    /*!
        Reflects trajectory of a uncharged particle.
    */
    class StraightTrajectory : public Trajectory
    {
        public:
        //! Constructor.
        /*
            \param particle A (usually uncharged) particle (or charged particle with very high pT so that trajectory can be considered straight).
        */
        StraightTrajectory(const Particle & particle) : Trajectory(particle) {;}

        //! Use Copy Constructor.
        /*
            \param trajectory StraightTrajectory does not have any special attribues so it can be copied right away
        */
        StraightTrajectory(const Trajectory & trajectory) : Trajectory(trajectory) {;}

        //! Check if an intersection of the trajectory with a barrel layer exists.
        /*!
            There is always an intersection between a straight line and a barrel layer unless the trajectory is parallel to z axis. In this case, the particle is not propagated anyways since it will not hit any detector material.
            \param layer A barrel layer.
            \return true
        */
        bool crosses(const BarrelSimplifiedGeometry & layer) const override {return true;}

        //! Return delta time (t*c) of the next intersection of trajectory and barrel layer
        /*!
            This function solves the quadratic equation (basically intersection a circle with a given radius and a straight line) in order to calculate the moment in time when the particle's trajectory intersects with a given barrel layer.
            \param layer A barrel layer.
            \param onLayer Specify if the particle already is on the layer (in that case the second solution has to be picked).
            \return t*c [ns * cm/ns] of next intersection (-1 if there is none).
        */
        double nextCrossingTimeC(const BarrelSimplifiedGeometry & layer, bool onLayer = false) const override;

        //! Move the particle along the helix trajectory for a given time.
        /*!
            \param deltaTimeC Time in units of t*c..
        */
        void move(double deltaTimeC) override;
    };
}

#endif
