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
#include <DataFormats/GeometrySurface/interface/BoundPlane.h>
#include <Geometry/CommonDetUnit/interface/GeomDetType.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <Geometry/CSCGeometry/interface/CSCChamberSpecs.h>

class CSCLayer;

class CSCChamber : public GeomDet {

public:

  CSCChamber( const BoundPlane::BoundPlanePointer& bp, CSCDetId id, const CSCChamberSpecs* specs ) :
  GeomDet( bp ),  theChamberSpecs( specs ), 
    theComponents(6,(const CSCLayer*)nullptr) {
    setDetId(id);
  }

  ~CSCChamber() override;

  const GeomDetType& type() const override { return *(specs()); }

  /// Get the (concrete) DetId.
  CSCDetId id() const { return geographicalId(); }

  // Which subdetector
  SubDetector subDetector() const override {return GeomDetEnumerators::CSC;}

  const CSCChamberSpecs* specs() const { return theChamberSpecs; }

  /// Return the layers in this chamber
  std::vector< const GeomDet* > components() const override;

  /// Return the layer with a given id in this chamber
  const GeomDet* component(DetId id) const override;


  // Extension of the interface

  /// Add a layer
  void addComponent( int n, const CSCLayer* gd );

  /// Return all layers
  const std::vector< const CSCLayer* >& layers() const { return theComponents; }

  /// Return the layer corresponding to the given id 
  const CSCLayer* layer(CSCDetId id) const;
  
  /// Return the given layer.
  /// Layers are numbered 1-6.
  const CSCLayer* layer(int ilay) const;

private:

  const CSCChamberSpecs* theChamberSpecs;
  std::vector< const CSCLayer* > theComponents; // the 6 CSCLayers comprising a CSCChamber; are owned by this class
};

#endif // Geometry_CSCGeometry_CSCChamber_H
