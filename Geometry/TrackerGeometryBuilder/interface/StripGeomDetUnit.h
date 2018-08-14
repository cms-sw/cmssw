#ifndef Geometry_TrackerGeometryBuilder_StripGeomDetUnit_H
#define Geometry_TrackerGeometryBuilder_StripGeomDetUnit_H

#include "Geometry/CommonDetUnit/interface/TrackerGeomDet.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/TrackerGeometryBuilder/interface/ProxyStripTopology.h"

class StripGeomDetType;
class StripTopology;
class SurfaceDeformation;
/**
 * StripGeomDetUnit is the abstract class for SiStripGeomDetUnit.
 */

class StripGeomDetUnit final : public TrackerGeomDet {
public:

  StripGeomDetUnit( BoundPlane* sp, StripGeomDetType const * type, DetId id);

  // Det interface

  /// NOTE (A.M.): The actual pointer to StripGeomDetType is now a member of the
  /// proxy topology. As StripGeomDetType has the actual topology as a pointer,
  /// it is possible to access this topology in two different ways. Once via
  /// the proxy topology (through topology() and specificTopology()) which includes
  /// corrections for the surface deformations, and once via the GeomDetType
  /// (through type().topology() and the like).
  const GeomDetType& type() const override;

  /// Returns a reference to the strip proxy topology
  const Topology& topology() const override;

  /// NOTE (A.M.): The actual pointer to StripGeomDetType is now a member of the
  /// proxy topology. As StripGeomDetType has the actual topology as a pointer,
  /// it is possible to access this topology in two different ways. Once via
  /// the proxy topology (through topology() and specificTopology()) which includes
  /// corrections for the surface deformations, and once via the GeomDetType
  /// (through type().topology() and the like).
  virtual StripGeomDetType const & specificType() const;

  /// Returns a reference to the strip proxy topology
  virtual const StripTopology& specificTopology() const;

  /// Return pointer to surface deformation.
  const SurfaceDeformation * surfaceDeformation() const override { 
    return theTopology->surfaceDeformation();
  }

  bool isLeaf()	const override	{ return true;}


private:

  /// set the SurfaceDeformation for this StripGeomDetUnit to proxy topology.
  void setSurfaceDeformation(const SurfaceDeformation * deformation) override;

  std::unique_ptr<ProxyStripTopology> theTopology;
};

#endif // Tracker_StripGeomDetUnit_H
