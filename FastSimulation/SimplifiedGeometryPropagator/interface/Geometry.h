#ifndef FASTSIM_GEOMETRY_H
#define FASTSIM_GEOMETRY_H

#include "DataFormats/Math/interface/LorentzVector.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/ForwardSimplifiedGeometry.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/BarrelSimplifiedGeometry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

///////////////////////////////////////////////
// Author: L. Vanelderen, S. Kurz
// Date: 29 May 2017
//////////////////////////////////////////////////////////

class GeometricSearchTracker;
class MagneticField;

#include <vector>

namespace edm {
  //class ParameterSet;
  class EventSetup;
}  // namespace edm

namespace fastsim {
  class InteractionModel;

  //! Definition the tracker geometry (vectors of forward/barrel layers).
  /*!
        This class models the material budget of the tracker. Those are reflected by 2 vectors of forward (disks, ordered by increasing Z-position) and barrel layers respectively (cylinders, ordered by increasing radius).
        Furthermore, initiatilizes the magnetic field used for propagation of particles inside the tracker.
    */
  class Geometry {
  public:
    //! Constructor.
    Geometry(const edm::ParameterSet& cfg);

    //! Default destructor.
    ~Geometry();

    //! Initializes the tracker geometry.
    /*!
            Calls SimplifiedGeometryFactory to initialize the vectors of barrel/forward layers and provides magnetic field and interaction models for those.
            \param iSetup The Event Setup.
            \param interactionModelMap Map of all interaction models considered (for any layer)
            \sa SimplifiedGeometryFactory
        */
    void update(const edm::EventSetup& iSetup, const std::map<std::string, InteractionModel*>& interactionModelMap);

    //! Initializes the tracker geometry.
    /*!
            Get the field from the MagneticFieldRecord (or set constant if defined in python config)
            \param position The position where you want to get the magnetic field (field only in Z direction).
            \return The magnetic field (Z-direction) for a given position.
        */
    double getMagneticFieldZ(const math::XYZTLorentzVector& position) const;

    //! Return the vector of barrel layers.
    /*!
            Ordered by increasing radius (0 to +inf).
            \return The barrel layers according to defined geometry.
        */
    const std::vector<std::unique_ptr<BarrelSimplifiedGeometry>>& barrelLayers() const { return barrelLayers_; }

    //! Return the vector of forward layers (disks).
    /*!
            Ordered by increasing Z-position (-inf to +inf).
            \return The forward layers according to defined geometry.
        */
    const std::vector<std::unique_ptr<ForwardSimplifiedGeometry>>& forwardLayers() const { return forwardLayers_; }

    //! Upper bound of the radius of the whole tracker geometry.
    /*!
            Necessary to initialize the magnetic field within this volume.
            \return Upper bound of radius of the whole tracker geometry
        */
    double getMaxRadius() { return maxRadius_; }

    //! Upper bound of the length/2 (0 to +Z) of the whole tracker geometry.
    /*!
            Necessary to initialize the magnetic field within this volume.
            \return Upper bound of length/2 of the whole tracker geometry
        */
    double getMaxZ() { return maxZ_; }

    //! Provides some basic output for e.g. debugging.
    friend std::ostream& operator<<(std::ostream& o, const fastsim::Geometry& geometry);

    //! Helps to navigate through the vector of barrel layers.
    /*!
            For a given layer, returns the next layer (as ordered in std::vector<...> barrelLayers_).
            \param layer A barrel layer
            \return The next layer (increasing radius). Returns 0 if last layer reached.
        */
    const BarrelSimplifiedGeometry* nextLayer(const BarrelSimplifiedGeometry* layer) const {
      if (layer == nullptr) {
        return nullptr;
      }
      unsigned nextLayerIndex = layer->index() + 1;
      return nextLayerIndex < barrelLayers_.size() ? barrelLayers_[nextLayerIndex].get() : nullptr;
    }

    //! Helps to navigate through the vector of forward layers.
    /*!
            For a given layer, returns the next layer (as ordered in std::vector<...> forwardLayers_).
            \param layer A forward layer
            \return The next layer (increasing Z-position). Returns 0 if last layer reached.
        */
    const ForwardSimplifiedGeometry* nextLayer(const ForwardSimplifiedGeometry* layer) const {
      if (layer == nullptr) {
        return nullptr;
      }
      unsigned nextLayerIndex = layer->index() + 1;
      return nextLayerIndex < forwardLayers_.size() ? forwardLayers_[nextLayerIndex].get() : nullptr;
    }

    //! Helps to navigate through the vector of barrel layers.
    /*!
            For a given layer, returns the previous layer (as ordered in std::vector<...> barrelLayers_).
            \param layer A barrel layer
            \return The previous layer (decreasing radius). Returns 0 if first layer reached.
        */
    const BarrelSimplifiedGeometry* previousLayer(const BarrelSimplifiedGeometry* layer) const {
      if (layer == nullptr) {
        return barrelLayers_.back().get();
      }
      return layer->index() > 0 ? barrelLayers_[layer->index() - 1].get() : nullptr;
    }

    //! Helps to navigate through the vector of forward layers.
    /*!
            For a given layer, returns the previous layer (as ordered in std::vector<...> forwardLayers_).
            \param layer A forward layer
            \return The previous layer (decreasing Z-position). Returns 0 if first layer reached.
        */
    const ForwardSimplifiedGeometry* previousLayer(const ForwardSimplifiedGeometry* layer) const {
      if (layer == nullptr) {
        return forwardLayers_.back().get();
      }
      return layer->index() > 0 ? forwardLayers_[layer->index() - 1].get() : nullptr;
    }

  private:
    std::vector<std::unique_ptr<BarrelSimplifiedGeometry>>
        barrelLayers_;  //!< The vector of barrel layers (increasing radius)
    std::vector<std::unique_ptr<ForwardSimplifiedGeometry>>
        forwardLayers_;  //!< The vector of forward layers (increasing Z-position)
    std::unique_ptr<MagneticField>
        ownedMagneticField_;  //!< Needed to create a uniform magnetic field if speciefied in config

    unsigned long long cacheIdentifierTrackerRecoGeometry_;  //!< Check interval of validity of the tracker geometry
    unsigned long long cacheIdentifierIdealMagneticField_;   //!< Check interval of validity of the magnetic field

    const GeometricSearchTracker* geometricSearchTracker_;  //! The tracker geometry
    const MagneticField* magneticField_;                    //!< The magnetic field
    const bool useFixedMagneticFieldZ_;  //!< Needed to create a uniform magnetic field if speciefied in config
    const double fixedMagneticFieldZ_;   //!< Use a uniform magnetic field or non-uniform from MagneticFieldRecord
    const bool
        useTrackerRecoGeometryRecord_;  //!< Use GeometricSearchTracker (active layers/reco geometry). Can be used to get position/radius of tracker material that reflects active layers
    const std::string trackerAlignmentLabel_;  //!< The tracker alignment label
    const std::vector<edm::ParameterSet>
        barrelLayerCfg_;  //!< The config in which all parameters of the barrel layers are defined
    const std::vector<edm::ParameterSet>
        forwardLayerCfg_;     //!< The config in which all parameters of the forward layers are defined
    const double maxRadius_;  //! Upper bound of the radius of the whole tracker geometry
    const double maxZ_;       //! Upper bound of the length/2 (0 to +Z) of the whole tracker geometry

    const bool barrelBoundary_;                          //!< Hack to interface "old" calo to "new" tracking
    const bool forwardBoundary_;                         //!< Hack to interface "old" calo to "new" tracking
    const edm::ParameterSet trackerBarrelBoundaryCfg_;   //!< Hack to interface "old" calo to "new" tracking
    const edm::ParameterSet trackerForwardBoundaryCfg_;  //!< Hack to interface "old" calo to "new" tracking
  };
  std::ostream& operator<<(std::ostream& os, const fastsim::Geometry& geometry);
}  // namespace fastsim

#endif
