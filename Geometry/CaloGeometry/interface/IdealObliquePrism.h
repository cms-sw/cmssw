#ifndef GEOMETRY_CALOGEOMETRY_IDEALOBLIQUEPRISM_H
#define GEOMETRY_CALOGEOMETRY_IDEALOBLIQUEPRISM_H 1

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

namespace calogeom {
  /** \class IdealObliquePrism
    
  Oblique prism class used for HCAL  (HB, HE, HO) volumes.
  
  Required parameters for an ideal oblique prism:
  
  - eta, phi of axis
  - radial distance (along axis) to front and back faces
  - single bit - faces parallel or perpendicular to z-axis
  - eta width and phi width of faces (same for front/back)
  
  Total: 6+1 parameters
  
  Internally, the "point of reference" is the center (eta/phi) of the
  front face of the prism.  Therefore, the only internally stored
  parameters are eta and phi widths, the axis tower thickness, and the
  parallel/perpendicular setting.  The parallel/perpendicular setting
  is encoded in the sign of the thickness.  (positive = parallel to
  z-axis, negative = perpendicular)

  $Date: $
  $Revision: $
  \author J. Mans - Minnesota
  */
  class IdealObliquePrism : public cms::CaloCellGeometry {
  public:
    IdealObliquePrism(const GlobalPoint& faceCenter, float widthEta, float widthPhi, float thickness, bool parallelToZaxis);
    IdealObliquePrism(float eta, float phi, float radialDistanceToFront, float widthEta, float widthPhi, float thickness, bool parallelToZaxis);
    virtual ~IdealObliquePrism() { }
    virtual bool inside(const GlobalPoint & point) const;  
    /// The corners in the oblique prism are stored transiently.
    virtual const std::vector<GlobalPoint> & getCorners() const;
  private:
    float hwidthEta_, hwidthPhi_; // half-widths
    float thickness_;
    mutable std::vector<GlobalPoint> points_; // required for now...  Maybe reorganized later for speed.
  };
}

#endif
