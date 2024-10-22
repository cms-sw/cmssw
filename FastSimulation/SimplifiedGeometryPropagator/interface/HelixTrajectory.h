
#ifndef FASTSIM_HELIXTRAJECTORY_H
#define FASTSIM_HELIXTRAJECTORY_H

#include "FastSimulation/SimplifiedGeometryPropagator/interface/Trajectory.h"

///////////////////////////////////////////////
// Author: L. Vanelderen, S. Kurz
// Date: 29 May 2017
//////////////////////////////////////////////////////////

namespace fastsim {
  //! Mathematical representation of a helix.
  /*!
        Reflects trajectory of a charged particle in a magnetic field.
        The trajectory is defined by cylindrical coordinates (see definition of variables for more information).
    */
  class HelixTrajectory : public Trajectory {
  public:
    //! Constructor.
    /*!
            The magnetic field is (to good approximation) constant between two tracker layers (and only in Z-direction).
            \param particle A (usually charged) particle.
            \param magneticFieldZ The magnetic field.
        */
    HelixTrajectory(const Particle& particle, double magneticFieldZ);

    //! Check if an intersection of the trajectory with a barrel layer exists.
    /*!
            \param layer A barrel layer.
        */
    bool crosses(const BarrelSimplifiedGeometry& layer) const override;

    //! Return delta time (t*c) of the next intersection of trajectory and barrel layer
    /*!
            This function solves the quadratic equation (basically intersection of two circles with a given radius)
            in order to calculate the moment in time when the particle's trajectory intersects with a given barrel layer.
            \param layer A barrel layer.
            \param onLayer Specify if the particle already is on the layer (in that case the second solution has to be picked).
            \return t*c [ns * cm/ns] of next intersection (-1 if there is none).
        */
    double nextCrossingTimeC(const BarrelSimplifiedGeometry& layer, bool onLayer = false) const override;

    //! Move the particle along the helix trajectory for a given time.
    /*!
            \param deltaTimeC Time in units of t*c..
        */
    void move(double deltaTimeC) override;

    //! Return distance of particle from center of the detector if it was at given angle phi of the helix
    /*!
            \param phi angle of the helix
        */
    double getRadParticle(double phi) const;

  private:
    const double radius_;  //!< The radius of the helix
    const double phi_;     //!< The angle of the particle alone the helix.
        //!< Ranges from 0 to 2PI: 0 corresponds to the positive X direction, phi increases counterclockwise
    const double centerX_;   //!< X-coordinate of the center of the helix
    const double centerY_;   //!< Y-coordinate of the center of the helix
    const double centerR_;   //!< Distance of the center of the helix from the center of the tracker
    const double minR_;      //!< The minimal distance of the helix from the center of the tracker
    const double maxR_;      //!< The maximum distance of the helix from the center of the tracker
    const double phiSpeed_;  //!< The angular speed of the particle on the helix trajectory
  };
}  // namespace fastsim

#endif
