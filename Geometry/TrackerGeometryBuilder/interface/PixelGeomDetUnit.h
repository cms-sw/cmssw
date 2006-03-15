#ifndef Geometry_TrackerGeometryBuilder_PixelGeomDetUnit_H
#define Geometry_TrackerGeometryBuilder_PixelGeomDetUnit_H


#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"


class PixelGeomDetType;
class PixelTopology;
class GeometricDet;
/**
 * The base PixelGeomDetUnit. Specialized in SiPixelGeomDetUnit.
 */

class PixelGeomDetUnit : public GeomDetUnit {
public:

  PixelGeomDetUnit(BoundPlane* sp, PixelGeomDetType*& type, const GeometricDet*& gd);

  // DetUnit interface

  virtual GeomDetType& type() const;

  virtual const Topology& topology() const;

  DetId geographicalId() const;

  virtual PixelGeomDetType& specificType() const { return *theType;}

  virtual const PixelTopology& specificTopology() const;

private:
  PixelGeomDetType* theType;
  const GeometricDet* theGD;
};

#endif // Tracker_PixelGeomDetUnit_H
