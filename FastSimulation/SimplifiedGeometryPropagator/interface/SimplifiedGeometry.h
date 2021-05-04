#ifndef FASTSIM_SIMPLIFIEDGEOMETRY_H
#define FASTSIM_SIMPLIFIEDGEOMETRY_H

#include "DataFormats/Math/interface/LorentzVector.h"

#include <memory>
#include <vector>

///////////////////////////////////////////////
// Author: L. Vanelderen, S. Kurz
// Date: 29 May 2017
//////////////////////////////////////////////////////////

class DetLayer;
class MagneticField;
class GeometricSearchTracker;
class TH1F;

namespace edm {
  class ParameterSet;
}

namespace fastsim {
  class InteractionModel;
  class SimplifiedGeometryFactory;

  //! Implementation of a generic detector layer (base class for forward/barrel layers).
  /*!
        Each layer has a geometric attribute ('geomProperty') which depends on which kind of layer is actually created
        (radius for barrel layers, position z for forward layers).  Furthermore, a thickness in radiation length is assigned
        which can vary throughout the layer.
        \sa BarrelSimplifiedGeometry
        \sa ForwardSimplifiedGeometry
    */
  class SimplifiedGeometry {
  public:
    //! Default constructor.
    /*!
            'geomProperty' depends on which kind of layer is actually created
            - BarrelSimplifiedGeometry: radius
            - ForwardSimplifiedGeometry: z
        */
    SimplifiedGeometry(double geomProperty);

    SimplifiedGeometry(SimplifiedGeometry&&) = default;
    SimplifiedGeometry(SimplifiedGeometry const&) = delete;

    SimplifiedGeometry& operator=(SimplifiedGeometry&&) = default;
    SimplifiedGeometry& operator=(SimplifiedGeometry const&) = delete;

    //! Default destructor.
    virtual ~SimplifiedGeometry();

    ////////
    // HACK
    ////////

    //! Hack to interface "old" Calorimetry with "new" Tracker
    enum CaloType { NONE, TRACKERBOUNDARY, PRESHOWER1, PRESHOWER2, ECAL, HCAL, VFCAL };

    //! Hack to interface "old" Calorimetry with "new" Tracker
    void setCaloType(CaloType type) { caloType_ = type; }

    //! Hack to interface "old" Calorimetry with "new" Tracker
    CaloType getCaloType() const { return caloType_; }

    ////////
    // END
    ////////

    //! Set index of this layer (layers are numbered according to their position in the detector).
    /*!
            The (usual) order is increasing 'geomProperty':
            - BarrelLayers: innermost to outermost
            - ForwardLayers: -z to +z
            Makes it more convenient to navigate from one layer to the previous/next layer.
        */
    void setIndex(int index) { index_ = index; }

    //! Return index of this layer (layers are numbered according to their position in the detector).
    /*!
            The (usual) order is increasing 'geomProperty':
            - BarrelLayers: innermost to outermost
            - ForwardLayers: -z to +z
            Makes it more convenient to navigate from one layer to the previous/next layer.
        */
    int index() const { return index_; }

    //! Return geometric attribute of the layer.
    /*!
            'geomProperty' depends on which kind of layer is actually created
            - BarrelSimplifiedGeometry: radius
            - ForwardSimplifiedGeometry: z
        */
    const double getGeomProperty() const { return geomProperty_; }

    //! Return thickness of the layer at a given position.
    /*!
            Returns the thickness of the layer (in radiation length) at a specified position since the thickness can vary throughout the layer.
            \param position A position which has to be on the layer.
            \return The thickness of the layer (in radiation length).
            \sa getThickness(const math::XYZTLorentzVector & position, const math::XYZTLorentzVector & momentum)
        */
    virtual const double getThickness(const math::XYZTLorentzVector& position) const = 0;

    //! Return thickness of the layer at a given position, also considering the incident angle.
    /*!
            Returns the thickness of the layer (in radiation length) at a specified position and a given incident angle since the thickness can vary throughout the layer.
            \param position A position which has to be on the layer.
            \param momentum The momentum of the incident particle.
            \return The thickness of the layer (in radiation length).
            \sa getThickness(const math::XYZTLorentzVector & position)
        */
    virtual const double getThickness(const math::XYZTLorentzVector& position,
                                      const math::XYZTLorentzVector& momentum) const = 0;

    //! Some layers have a different thickness for nuclear interactions.
    /*!
            Right now only considered for TEC layers.
        */
    const double getNuclearInteractionThicknessFactor() const { return nuclearInteractionThicknessFactor_; }

    //! Return pointer to the assigned active layer (if any).
    /*!
            Necessary to create SimHits.
        */
    const DetLayer* getDetLayer() const { return detLayer_; }

    //! Return magnetic field (field only has Z component!) on the layer.
    /*!
            \param position A position which has to be on the layer.
            \return The magnetic field on the layer.
        */
    virtual const double getMagneticFieldZ(const math::XYZTLorentzVector& position) const = 0;

    //! Returns false/true depending if the object is a (non-abstract) barrel/forward layer.
    /*!
            Function to easily destinguish barrel from forward layers (which both inherit from SimplifiedGeometry).
        */
    virtual bool isForward() const = 0;

    //! Return the vector of all interaction models that are assigned with a layer.
    /*!
            This makes it easy to switch on/off some interactions for some layers.
        */
    const std::vector<InteractionModel*>& getInteractionModels() const { return interactionModels_; }

    //! Some basic output.
    friend std::ostream& operator<<(std::ostream& os, const SimplifiedGeometry& layer);

    friend class fastsim::SimplifiedGeometryFactory;

  protected:
    double geomProperty_;  //!< Geometric property of the layer: radius (barrel layer) / position z (forward layer)
    int index_;  //!< Return index of this layer (layers are numbered according to their position in the detector). The (usual) order is increasing 'geomProperty'.
    const DetLayer* detLayer_;                 //!< Return pointer to the assigned active layer (if any).
    std::unique_ptr<TH1F> magneticFieldHist_;  //!< Histogram that stores the size of the magnetic field along the layer.
    std::unique_ptr<TH1F> thicknessHist_;      //!< Histogram that stores the tickness (radLengths) along the layer.
    double nuclearInteractionThicknessFactor_;  //!< Some layers have a different thickness for nuclear interactions.
    std::vector<InteractionModel*>
        interactionModels_;  //!< Vector of all interaction models that are assigned with a layer.
    CaloType caloType_;      //!< Hack to interface "old" Calorimetry with "new" Tracker
  };

  std::ostream& operator<<(std::ostream& os, const SimplifiedGeometry& layer);

}  // namespace fastsim

#endif
