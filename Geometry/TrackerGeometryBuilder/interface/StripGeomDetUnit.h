#ifndef Geometry_TrackerGeometryBuilder_StripGeomDetUnit_H
#define Geometry_TrackerGeometryBuilder_StripGeomDetUnit_H

#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"

class StripGeomDetType;
class StripTopology;
class GeometricDet;
class SurfaceDeformation;
/**
 * StripGeomDetUnit is the abstract class for SiStripGeomDetUnit.
 */

class StripGeomDetUnit : public GeomDetUnit {
public:

  StripGeomDetUnit( BoundPlane* sp, StripGeomDetType* type,const GeometricDet* gd);

  // Det interface

  virtual const GeomDetType& type() const;

  virtual const Topology& topology() const;

  virtual StripGeomDetType& specificType() const { return *theType;}

  virtual const StripTopology& specificTopology() const;

  /// Return pointer to surface deformation.
  /// NOTE (A.M.): The actual surface deformation object being a member of
  /// StripGeomDetUnit is only temporary. Eventually it will move to a dedicated
  /// proxy topology class which will become a member of StripGeomDetUnit.
  virtual const SurfaceDeformation * surfaceDeformation() const { 
    return theSurfaceDeformation.operator->();
  }

private:

  /// set the SurfaceDeformation for this StripGeomDetUnit. PixelGeomDetUnit
  /// takes over ownership of SurfaceDeformation.
  /// NOTE (A.M.): The actual surface deformation object being a member of
  /// StripGeomDetUnit is only temporary. Eventually it will move to a dedicated
  /// proxy topology class which will become a member of StripGeomDetUnit.
  virtual void setSurfaceDeformation(const SurfaceDeformation * deformation);

  StripGeomDetType* theType;
  const GeometricDet* theGD;
  DeepCopyPointerByClone<const SurfaceDeformation> theSurfaceDeformation;
};

#endif // Tracker_StripGeomDetUnit_H
