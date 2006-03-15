#ifndef Geometry_TrackerGeometryBuilder_StripGeomDetUnit_H
#define Geometry_TrackerGeometryBuilder_StripGeomDetUnit_H

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"

class StripGeomDetType;
class StripTopology;
class GeometricDet;
/**
 * StripGeomDetUnit is the abstract class for SiStripGeomDetUnit.
 */

class StripGeomDetUnit : public GeomDetUnit {
public:

  StripGeomDetUnit( BoundPlane* sp, StripGeomDetType*& type,const GeometricDet*& gd);

  // Det interface

  virtual GeomDetType& type() const;

  virtual const Topology& topology() const;

  DetId geographicalId() const;

  virtual StripGeomDetType& specificType() const { return *theType;}

  virtual const StripTopology& specificTopology() const;


private:
  StripGeomDetType* theType;
  const GeometricDet* theGD;
};

#endif // Tracker_StripGeomDetUnit_H
