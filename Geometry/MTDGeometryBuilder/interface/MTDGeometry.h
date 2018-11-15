#ifndef Geometry_MTDGeometryBuilder_MTDGeometry_H
#define Geometry_MTDGeometryBuilder_MTDGeometry_H

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"
#include "Geometry/CommonDetUnit/interface/MTDGeomDet.h"

class GeometricTimingDet;

/**
 * A specific MTD Builder which builds MTD from a list of DetUnits. 
 * Pattern recognition is used to discover layers, rings etc.
 */
class MTDGeometry final : public TrackingGeometry {
  
public:
  using SubDetector = GeomDetEnumerators::SubDetector;
  
  enum class ModuleType {
    UNKNOWN, 
      BTL, 
      ETL, 
   };

  ~MTDGeometry() override ;

  const DetTypeContainer&  detTypes()         const override {return theDetTypes;}
  const DetContainer&      detUnits()         const override {return theDetUnits;}
  const DetContainer&      dets()             const override {return theDets;}
  const DetIdContainer&    detUnitIds()       const override {return theDetUnitIds;}
  const DetIdContainer&    detIds()           const override {return theDetIds;}
  const MTDGeomDet*        idToDetUnit(DetId) const override;
  const MTDGeomDet*        idToDet(DetId)     const override;

  const GeomDetEnumerators::SubDetector geomDetSubDetector(int subdet) const;
  unsigned int numberOfLayers(int subdet) const;
  bool isThere(GeomDetEnumerators::SubDetector subdet) const;

  unsigned int offsetDU(unsigned sid) const { return theOffsetDU[sid];}
  unsigned int endsetDU(unsigned sid) const { return theEndsetDU[sid];}
  // Magic : better be called at the right moment...
  void setOffsetDU(unsigned sid) { theOffsetDU[sid]=detUnits().size();}
  void setEndsetDU(unsigned sid) { theEndsetDU[sid]=detUnits().size();}
  void fillTestMap(const GeometricTimingDet* gd);

  ModuleType moduleType(const std::string& name) const;

  GeometricTimingDet const * trackerDet() const {return  theTrackerDet;}

  const DetContainer& detsBTL() const;
  const DetContainer& detsETL() const;
  
  ModuleType getDetectorType(DetId) const;
  float getDetectorThickness(DetId) const;


private:

  explicit MTDGeometry(GeometricTimingDet const* gd=nullptr);  
  
  friend class MTDGeomBuilderFromGeometricTimingDet;

  void addType(GeomDetType const * p);
  void addDetUnit(GeomDet const * p);
  void addDetUnitId(DetId p);
  void addDet(GeomDet const * p);
  void addDetId(DetId p);
  void finalize();

  GeometricTimingDet const * theTrackerDet; 

  /// Aligner has access to map
  friend class GeometryAligner;

  DetTypeContainer  theDetTypes;  // owns the DetTypes
  DetContainer      theDetUnits;  // they're all also into 'theDets', so we assume 'theDets' owns them
  unsigned int      theOffsetDU[2]; // offsets in the above
  unsigned int      theEndsetDU[2]; // end offsets in the above
  DetContainer      theDets;      // owns *ONLY* the GeomDet * corresponding to GluedDets.
  DetIdContainer    theDetUnitIds;
  DetIdContainer    theDetIds; 
  mapIdToDetUnit    theMapUnit; // does not own GeomDetUnit *
  mapIdToDet        theMap;     // does not own GeomDet *

  DetContainer      theBTLDets; // not owned: they're also in 'theDets'
  DetContainer      theETLDets; // not owned: they're also in 'theDets'
  
  GeomDetEnumerators::SubDetector theSubDetTypeMap[2];
  unsigned int theNumberOfLayers[2];
  std::vector< std::tuple< DetId, MTDGeometry::ModuleType, float> > theDetTypetList; 
};

#endif
