#ifndef Geometry_MTDGeometryBuilder_MTDGeomDetUnit_H
#define Geometry_MTDGeometryBuilder_MTDGeomDetUnit_H

#include "Geometry/CommonDetUnit/interface/MTDGeomDet.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"

class MTDGeomDetType;
class PixelTopology;
class SurfaceDeformation;
/**
 * The base PixelGeomDetUnit. Specialized in SiPixelGeomDetUnit.
 */

class MTDGeomDetUnit final : public MTDGeomDet {
public:

  MTDGeomDetUnit(BoundPlane* sp, MTDGeomDetType const * type, DetId id);

  // DetUnit interface

  /// NOTE (A.M.): The actual pointer to PixelGeomDetType is now a member of the
  /// proxy topology. As PixelGeomDetType has the actual topology as a pointer,
  /// it is possible to access this topology in two different ways. Once via
  /// the proxy topology (through topology() and specificTopology()) which includes
  /// corrections for the surface deformations, and once via the GeomDetType
  /// (through type().topology() and the like).
  const GeomDetType& type() const override;
  
  /// Returns a reference to the pixel proxy topology
  const Topology& topology() const override;

  /// NOTE (A.M.): The actual pointer to PixelGeomDetType is now a member of the
  /// proxy topology. As PixelGeomDetType has the actual topology as a pointer,
  /// it is possible to access this topology in two different ways. Once via
  /// the proxy topology (through topology() and specificTopology()) which includes
  /// corrections for the surface deformations, and once via the GeomDetType
  /// (through type().topology() and the like).
  virtual const MTDGeomDetType& specificType() const;

  /// Returns a reference to the pixel proxy topology
  virtual const PixelTopology& specificTopology() const;

  /// Return pointer to surface deformation.
  const SurfaceDeformation * surfaceDeformation() const override { 
    return theTopology->surfaceDeformation();
  }

  bool isLeaf()	const override	{ return true;}

private:

  /// set the SurfaceDeformation for this StripGeomDetUnit to proxy topology.
  void setSurfaceDeformation(const SurfaceDeformation * deformation) override;

  std::unique_ptr<ProxyMTDTopology> theTopology;
};

#endif // MTD_MTDGeomDetUnit_H
