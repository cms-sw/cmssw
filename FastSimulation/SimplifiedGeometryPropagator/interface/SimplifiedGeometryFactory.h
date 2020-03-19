#ifndef FASTSIM_SIMPLIFIEDGEOMETRYFACTORY
#define FASTSIM_SIMPLIFIEDGEOMETRYFACTORY

class GeometricSearchTracker;
class MagneticField;
class DetLayer;
class BarrelDetLayer;
class ForwardDetLayer;

#include <memory>
#include <map>
#include <vector>
#include <string>

///////////////////////////////////////////////
// Author: L. Vanelderen
// Date: 13 May 2014
//
// Revision: Class structure modified to match SimplifiedGeometryPropagator
//           S. Kurz, 29 May 2017
//////////////////////////////////////////////////////////

namespace edm {
  class ParameterSet;
}

namespace fastsim {
  class SimplifiedGeometry;
  class BarrelSimplifiedGeometry;
  class ForwardSimplifiedGeometry;
  class InteractionModel;

  //! Constructs a tracker layer according to entry in python config (incl interaction models).
  /*!
        Also creates links to DetLayer (if active layer) and stores strength of magnetic field along the layer.
        If some parameters are not stored in the config file, tries to get them from the full detector geometry (GeometricSearchTracker). This is however only possible for active layers.
        \sa Geometry()
        \sa SimplifiedGeometry()
    */
  class SimplifiedGeometryFactory {
  public:
    //! Constructor
    /*!
            \param geometricSearchTracker The full tracker geometry (needed for links to active detLayers).
            \param magneticField The full magnetic field.
            \param interactionModelMap Map of interaction models that should be assigned for that layer.
            \param magneticFieldHistMaxR Max Radius for initialization of magnetic field histogram (TH1, limit of axis).
            \param magneticFieldHistMaxZ Max Z for initialization of magnetic field histogram (TH1, limit of axis).
        */
    SimplifiedGeometryFactory(const GeometricSearchTracker *geometricSearchTracker,
                              const MagneticField &magneticField,
                              const std::map<std::string, fastsim::InteractionModel *> &interactionModelMap,
                              double magneticFieldHistMaxR,
                              double magneticFieldHistMaxZ);

    //! Each layer is either a barrel layer, or a forward layer (either at ppositive or negative Z).
    enum LayerType { BARREL, POSFWD, NEGFWD };

    //! Main method of this class. Creates a new detector layer (SimplifiedGeometry).
    /*!
            Reads the config file, does all the initialization etc. and creates either a forward or a barrel layer (depends on LayerType type).
            \param type Either BARREL, POSFWD or NEGFWD.
            \return A SimplifiedGeometry (either ForwardSimplifiedGeometry or BarrelSimplifiedGeometry).
        */
    std::unique_ptr<SimplifiedGeometry> createSimplifiedGeometry(LayerType type, const edm::ParameterSet &cfg) const;

    //! Helper method for createSimplifiedGeometry(..) to create a forward layer (ForwardSimplifiedGeometry).
    /*!
            \param type Either POSFWD or NEGFWD.
            \return A ForwardSimplifiedGeometry
            \sa createSimplifiedGeometry(LayerType type, const edm::ParameterSet & cfg)
        */
    std::unique_ptr<ForwardSimplifiedGeometry> createForwardSimplifiedGeometry(LayerType type,
                                                                               const edm::ParameterSet &cfg) const;

    //! Helper method for createSimplifiedGeometry(..) to create a barrel layer (BarrelSimplifiedGeometry).
    /*!
            \return A BarrelSimplifiedGeometry
            \sa createSimplifiedGeometry(LayerType type, const edm::ParameterSet & cfg)
        */
    std::unique_ptr<BarrelSimplifiedGeometry> createBarrelSimplifiedGeometry(const edm::ParameterSet &cfg) const;

  private:
    //! Method returns a pointer to a DetLayer according to the string that was passed.
    /*!
            A convention for the name of the layer is used.
            For barrel layers this is "XXX?" where XXX is a part of the tracker and ? is the index of the layer (starting at one).
            For forward layers one has to add neg/pos in front to distinguish between the disk at -Z and +Z spatial position, so the convention is "xxxXXX?"
            Valid names: BPix, TIB, TOB, negFPix, posFPix, negTID, posTID, negTEC, posTEC
            Accordingly, the innermost layer of the barrel pixel detector is "BPix1".
            \param detLayerName A string following the naming convention.
        */
    const DetLayer *getDetLayer(const std::string &detLayerName,
                                const GeometricSearchTracker &geometricSearchTracker) const;

    const GeometricSearchTracker *const geometricSearchTracker_;                     //!< The full tracker geometry.
    const MagneticField *const magneticField_;                                       //!< The full magnetic field.
    const std::map<std::string, fastsim::InteractionModel *> *interactionModelMap_;  //!< Map of interaction models.
    const double magneticFieldHistMaxR_;  //!< Limit in R for histogram of magnetic field.
    const double magneticFieldHistMaxZ_;  //!< Limit in +-Z for histogram of magnetic field.
    std::map<std::string, const std::vector<BarrelDetLayer const *> *>
        barrelDetLayersMap_;  //!< A map of strings and pointers to detLayers.
    std::map<std::string, const std::vector<ForwardDetLayer const *> *>
        forwardDetLayersMap_;  //!< A map of strings and pointers to detLayers.
  };
}  // namespace fastsim

#endif
