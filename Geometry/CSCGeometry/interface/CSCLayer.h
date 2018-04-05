#ifndef Geometry_CSCGeometry_CSCLayer_H
#define Geometry_CSCGeometry_CSCLayer_H

/** \class CSCLayer
 *
 * Describes the geometry of the lowest-level detector unit 
 * modelled by a C++ object in the endcap muon CSC system.
 *
 * \author Tim Cox
 */

#include <DataFormats/MuonDetId/interface/CSCDetId.h>

#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <Geometry/CommonDetUnit/interface/GeomDetType.h>
#include <Geometry/CSCGeometry/interface/CSCLayerGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCChamber.h>
#include <DataFormats/GeometryVector/interface/GlobalPoint.h>
#include <DataFormats/GeometrySurface/interface/BoundPlane.h>

//#include <boost/shared_ptr.hpp>
//typedef boost::shared_ptr<CSCChamber> Pointer2Chamber;

class CSCLayer : public GeomDetUnit {

public:

  CSCLayer( const BoundPlane::BoundPlanePointer& sp, CSCDetId id, const CSCChamber* ch, const CSCLayerGeometry* geo ) : 
  GeomDetUnit( sp ), theId( id ), theChamber( ch ), theGeometry( geo ) {
    setDetId(id);
}

  const GeomDetType& type() const override { return chamber()->type(); }

  const Topology& topology() const override { return *(geometry()->topology()); }


  /**
   * Get the (concrete) DetId
   */
 
  CSCDetId id() const { return theId; }

  /**
   * Access to object handling layer geomerty
   */
  const CSCLayerGeometry* geometry() const { return theGeometry; }

  /**
   * Access to parent chamber
   */
  const CSCChamber* chamber() const { return theChamber; }
  
  /**
   * Global point at center of the given strip,
   * Must be in CSCLayer so it can return global coordinates.
   */
  GlobalPoint centerOfStrip(int strip) const;

  /** 
   * Global point at centre of the given wire group.
   * Must be in CSCLayer so it can return global coordinates.
   */
  GlobalPoint centerOfWireGroup(int wireGroup) const;

private:

  CSCDetId theId;

  const CSCChamber* theChamber; // NOT owned
  // Pointer2Chamber theChamber; // use a smart pointer instead

  // Local geometry is handled by the LayerGeometry
  // but only the Layer itself knows how to transform to the 
  // global frame so global calculations are handled by the
  // Layer not the LayerGeometry.
  const CSCLayerGeometry* theGeometry; // must have topology()
};

#endif // Geometry_CSCGeometry_CSCLayer_H
