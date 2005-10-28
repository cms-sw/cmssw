#ifndef GEOMETRY_CALOGEOMETRY_IDEALZPRISM_H
#define GEOMETRY_CALOGEOMETRY_IDEALZPRISM_H 1

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

namespace calogeom {
  /** \class IdealZPrism
    
  Prism class used for HF volumes.  HF volumes are prisms with axes along the Z direction whose
  face shapes are set by 
  
  Required parameters for an ideal Z prism:
  
  - eta, phi of axis
  - Z location of front and back faces
  - eta width and phi width of frontface
  
  Total: 6 parameters
  
  Internally, the "point of reference" is the center (eta/phi) of the
  front face of the prism.  Therefore, the only internally stored
  parameters are eta and phi widths and the tower z thickness.

  $Date: 2005/10/03 22:35:23 $
  $Revision: $
  \author J. Mans - Minnesota
  */
  class IdealZPrism : public CaloCellGeometry {
  public:
    IdealZPrism(const GlobalPoint& faceCenter, float widthEta, float widthPhi, float deltaZ);
    IdealZPrism(float eta, float phi, float radialDistanceToFront, float widthEta, float widthPhi, float deltaZ);
    virtual ~IdealZPrism() { }
    virtual bool inside(const GlobalPoint & point) const;  
    /// The corners in the oblique prism are stored transiently.
    virtual const std::vector<GlobalPoint> & getCorners() const;
  private:
    float hwidthEta_, hwidthPhi_; // half-widths
    float deltaZ_;
    mutable std::vector<GlobalPoint> points_; // required for now...  Maybe reorganized later for speed.
  };
}

#endif
