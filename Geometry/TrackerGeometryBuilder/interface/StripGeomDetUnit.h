#ifndef Geometry_TrackerGeometryBuilder_StripGeomDetUnit_H
#define Geometry_TrackerGeometryBuilder_StripGeomDetUnit_H

#include <boost/shared_ptr.hpp>

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/TrackerTopology/interface/ProxyStripTopology.h"

class StripGeomDetType;
class StripTopology;
class GeometricDet;
class SurfaceDeformation;
/**
 * StripGeomDetUnit is the abstract class for SiStripGeomDetUnit.
 */

class StripGeomDetUnit : public GeomDetUnit {
public:

  StripGeomDetUnit( BoundPlane* sp, StripGeomDetType* type, const GeometricDet* gd);

  // Det interface

  /// NOTE (A.M.): The actual pointer to StripGeomDetType is now a member of the
  /// proxy topology. As StripGeomDetType has the actual topology as a pointer,
  /// it is possible to access this topology in two different ways. Once via
  /// the proxy topology (through topology() and specificTopology()) which includes
  /// corrections for the surface deformations, and once via the GeomDetType
  /// (through type().topology() and the like).
  virtual const GeomDetType& type() const;

  /// Returns a reference to the strip proxy topology
  virtual const Topology& topology() const;

  /// NOTE (A.M.): The actual pointer to StripGeomDetType is now a member of the
  /// proxy topology. As StripGeomDetType has the actual topology as a pointer,
  /// it is possible to access this topology in two different ways. Once via
  /// the proxy topology (through topology() and specificTopology()) which includes
  /// corrections for the surface deformations, and once via the GeomDetType
  /// (through type().topology() and the like).
  virtual StripGeomDetType& specificType() const;

  /// Returns a reference to the strip proxy topology
  virtual const StripTopology& specificTopology() const;

  /// Return pointer to surface deformation.
  virtual const SurfaceDeformation * surfaceDeformation() const { 
    return theTopology->surfaceDeformation();
  }

private:

  /// set the SurfaceDeformation for this StripGeomDetUnit to proxy topology.
  virtual void setSurfaceDeformation(const SurfaceDeformation * deformation);

  boost::shared_ptr<ProxyStripTopology> theTopology;
  const GeometricDet* theGD;
};

#endif // Tracker_StripGeomDetUnit_H
