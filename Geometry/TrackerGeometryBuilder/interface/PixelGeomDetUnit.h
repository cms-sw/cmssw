#ifndef Geometry_TrackerGeometryBuilder_PixelGeomDetUnit_H
#define Geometry_TrackerGeometryBuilder_PixelGeomDetUnit_H

#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"

class PixelGeomDetType;
class PixelTopology;
class GeometricDet;
class SurfaceDeformation;
/**
 * The base PixelGeomDetUnit. Specialized in SiPixelGeomDetUnit.
 */

class PixelGeomDetUnit : public GeomDetUnit {
public:

  PixelGeomDetUnit(BoundPlane* sp, PixelGeomDetType* type, const GeometricDet* gd);

  // DetUnit interface

  virtual const GeomDetType& type() const;

  virtual const Topology& topology() const;

  virtual PixelGeomDetType& specificType() const { return *theType;}

  virtual const PixelTopology& specificTopology() const;

  /// Return pointer to surface deformation.
  /// NOTE (A.M.): The actual surface deformation object being a member of
  /// PixelGeomDetUnit is only temporary. Eventually it will move to a dedicated
  /// proxy topology class which will be a member of PixelGeomDetUnit.
  const SurfaceDeformation * surfaceDeformation() const { 
    return theSurfaceDeformation.operator->();
  }

private:

  /// set the SurfaceDeformation for this StripGeomDetUnit.
  /// NOTE (A.M.): The actual surface deformation object being a member of
  /// StripGeomDetUnit is only temporary. Eventually it will move to a dedicated
  /// proxy topology class which will be a member of PixelGeomDetUnit.
  virtual void setSurfaceDeformation(const SurfaceDeformation * deformation);

  PixelGeomDetType* theType;
  const GeometricDet* theGD;
  DeepCopyPointerByClone<const SurfaceDeformation> theSurfaceDeformation;
};

#endif // Tracker_PixelGeomDetUnit_H
