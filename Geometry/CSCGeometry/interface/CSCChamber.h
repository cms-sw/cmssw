#ifndef Geometry_CSCGeometry_CSCChamber_H
#define Geometry_CSCGeometry_CSCChamber_H

/** \class CSCChamber
 *
 * Describes the geometry of the second-level detector unit 
 * modelled by a C++ object in the endcap muon CSC system.
 * A CSCChamber is composed of 6 CSCLayer's and is,
 * of course, a Cathode Strip Chamber Chamber!
 *
 * \author Tim Cox
 */

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <Geometry/CommonDetUnit/interface/GeomDetType.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <Geometry/CSCGeometry/interface/CSCChamberSpecs.h>


class CSCChamber : public GeomDet {

public:

  CSCChamber( BoundPlane* bp, CSCDetId id, CSCChamberSpecs* specs ) :
  GeomDet( bp ), theId( id ), theChamberSpecs( specs ), 
  theComponents( std::vector< const GeomDet* >() ) {}

  const GeomDetType& type() const { return *(specs()); }

  DetId geographicalId() const { return theId; } //@@ Slices base

  CSCDetId cscId() const { return theId; }

  const CSCChamberSpecs* specs() const { return theChamberSpecs; }

  void addComponent( int n, const GeomDet* gd ) { theComponents.push_back( gd ); }

  virtual std::vector< const GeomDet* > components() const { return theComponents; }

private:

  CSCDetId theId;
  CSCChamberSpecs* theChamberSpecs;
  std::vector< const GeomDet* > theComponents; // the 6 CSCLayers comprising a CSCChamber
};

#endif // Geometry_CSCGeometry_CSCChamber_H
