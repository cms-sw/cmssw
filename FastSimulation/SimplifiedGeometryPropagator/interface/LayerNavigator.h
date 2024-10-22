#ifndef FASTSIM_LAYERNAVIGATOR_H
#define FASTSIM_LAYERNAVIGATOR_H

#include <string>

///////////////////////////////////////////////
// Author: L. Vanelderen, S. Kurz
// Date: 29 May 2017
//////////////////////////////////////////////////////////

namespace fastsim {
  class SimplifiedGeometry;
  class ForwardSimplifiedGeometry;
  class BarrelSimplifiedGeometry;
  class Geometry;
  class Particle;

  //! Handles/tracks (possible) intersections of particle's trajectory and tracker layers.
  /*!
        the geometry is described by 2 sets of layers:
            - forward layers: 
                flat layers, perpendicular to the z-axis, positioned at a given z
                these layers have material / instruments between a given materialMinR and materialMaxR
                no 2 forward layers should have the same z-position
            - barrel layers: 
                cylindrically shaped layers, with the z-axis as axis, infinitely long
                these layers have material / instruments for |z| < materialMaxAbsZ
                no 2 barrel layers should have the same radius
            - forward (barrel) layers are ordered according to increasing z (r)

        principle
            - neutral particles follow a straight trajectory
            - charged particles follow a helix-shaped trajectory:
                constant speed along the z-axis
                circular path in the x-y plane
            => the next layer that the particle will cross is among the following 3 layers
                - closest forward layer with
                - z >(<) particle.z() for particles moving in the positive(negative) direction
                - closest barrel layer with r < particle.r
                - closest barrel layer with r > particle.r  

        algorithm
            - find the 3 candidate layers 
            - find the earliest positive intersection time for each of the 3 candidate layers
            - move the particle to the earliest intersection time
            - select and return the layer with the earliest positive intersection time
    */
  class LayerNavigator {
  public:
    //! Constructor.
    /*!
            \param geometry The geometry of the tracker material.
        */
    LayerNavigator(const Geometry& geometry);

    //! Move particle along its trajectory to the next intersection with any of the tracker layers.
    /*!
            \param particle The particle that has to be moved to the next layer.
            \param layer The layer to which the particle was moved in the previous call of this function (0 if first call). Returns the layer this particle was then moved to.
            \return true / false if propagation succeeded / failed.
        */
    bool moveParticleToNextLayer(Particle& particle, const SimplifiedGeometry*& layer);

  private:
    const Geometry* const geometry_;  //!< The geometry of the tracker material
    const BarrelSimplifiedGeometry*
        nextBarrelLayer_;  //!< Pointer to the next (direction of the particle's momentum) barrel layer
    const BarrelSimplifiedGeometry*
        previousBarrelLayer_;  //!< Pointer to the previous (opposite direction of the particle's momentum) barrel layer
    const ForwardSimplifiedGeometry*
        nextForwardLayer_;  //!< Pointer to the next (direction of the particle's momentum) forward layer
    const ForwardSimplifiedGeometry*
        previousForwardLayer_;  //!< Pointer to the previous (opposite direction of the particle's momentum) forward layer
    static const std::string MESSAGECATEGORY;
  };
}  // namespace fastsim

#endif
