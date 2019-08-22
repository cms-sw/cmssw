#ifndef FASTSIM_TRAJECTORY_H
#define FASTSIM_TRAJECTORY_H

#include <memory>

#include "DataFormats/Math/interface/LorentzVector.h"

///////////////////////////////////////////////
// Author: L. Vanelderen, S. Kurz
// Date: 29 May 2017
//////////////////////////////////////////////////////////

namespace fastsim {
  class SimplifiedGeometry;
  class BarrelSimplifiedGeometry;
  class ForwardSimplifiedGeometry;
  class Particle;

  //! Definition the generic trajectory of a particle (base class for helix/straight trajectories).
  /*!
        Mathematical representation of a particle's trajectory. Provides three basic funtions:
        - Check if an intersection between a trajectory and a layer exists
        - Get the time when this happens (delta time in units of t*c)
        - Move the particle along it's trajectory for a given time (delta time in units of t*c)
    */
  class Trajectory {
  public:
    //! Calls constructor of derived classes.
    /*!
            Decides whether a straight (uncharged particle or very high pT charged particle) or a helix trajectory (generic charged particle) is constructed.
            \param particle The particle that should be decayed.
            \param magneticFieldZ The strenght of the magnetic field at the position of the particle.
            \return A StraightTrajectory or a HelixTrajectory
        */
    static std::unique_ptr<Trajectory> createTrajectory(const fastsim::Particle& particle, const double magneticFieldZ);

    //! Check if trajectory crosses a barrel layer.
    /*!
            Virtual function since different behavior of straight and helix trajectory.
            This funtion does not exist for forward layers since those are always hit - unless particle has exactly 0 momentum in Z direction which doesn't happen for numerical reasons.
        */
    virtual bool crosses(const BarrelSimplifiedGeometry& layer) const = 0;

    //! Simple getter: return position of the particle that was used to create trajectory.
    const math::XYZTLorentzVector& getPosition() { return position_; }

    //! Simple getter: return momentum of the particle that was used to create trajectory.
    const math::XYZTLorentzVector& getMomentum() { return momentum_; }

    //! Return delta time (t*c) of the next intersection of trajectory and generic layer
    /*!
            Calculation different for barrel/forward layers and straight/helix trajectory. Chooses which function has to be called.
            \param layer A barrel or forward layer.
            \param onLayer Specify if the particle already is on the layer (leads to different constraints for forward/barrel layers).
            \return t*c [ns * cm/ns] of next intersection (-1 if there is none).
        */
    double nextCrossingTimeC(const SimplifiedGeometry& layer, bool onLayer = false) const;

    //! Return delta time (t*c) of the next intersection of trajectory and forward layer
    /*!
            Since only momentum in Z direction matters, same function for straight and helix trajectories.
            \param layer A forward layer.
            \param onLayer Specify if the particle already is on the layer (in this case there is no solution).
            \return t*c [ns * cm/ns] of next intersection (-1 if there is none).
        */
    double nextCrossingTimeC(const ForwardSimplifiedGeometry& layer, bool onLayer = false) const;

    //! Return delta time (t*c) of the next intersection of trajectory and barrel layer
    /*!
            Different treatment of intersection of straight/helix layers with barrel layers. Implementation in derived classes.
            \param layer A barrel layer.
            \param onLayer Specify if the particle already is on the layer (in that case the second solution has to be picked).
            \return t*c [ns * cm/ns] of next intersection (-1 if there is none).
        */
    virtual double nextCrossingTimeC(const BarrelSimplifiedGeometry& layer, bool onLayer = false) const = 0;

    //! Move the particle along the trajectory for a given time.
    /*!
            \param deltaTimeC Time in units of t*c..
        */
    virtual void move(double deltaTimeC) = 0;
    virtual ~Trajectory();

  protected:
    //! Constructor of base class.
    /*!
            Usually constructor of derived classes HelixTrajectory or StraightTrajectory are called.
        */
    Trajectory(const fastsim::Particle& particle);

    math::XYZTLorentzVector position_;  //!< position of the particle that was used to create trajectory
    math::XYZTLorentzVector momentum_;  //!< momentum of the particle that was used to create trajectory
  };
}  // namespace fastsim

#endif
