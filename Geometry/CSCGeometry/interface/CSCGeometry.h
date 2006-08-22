#ifndef CSCGeometry_CSCGeometry_h
#define CSCGeometry_CSCGeometry_h

/** \class CSCGeometry
 *
 *  The model of the geometry of the endcap muon CSC detectors.
 *
 *  \author Tim Cox
 */

#include <DataFormats/DetId/interface/DetId.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <Geometry/CommonDetUnit/interface/TrackingGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <vector>
#include <map>

class GeomDetType;
class GeomDetUnit;
class CSCChamber;


class CSCGeometry : public TrackingGeometry {

  typedef std::map<DetId, GeomDet*> CSCDetMap;
  typedef std::vector<CSCChamber*> ChamberContainer;
  typedef std::vector<CSCLayer*> LayerContainer;

 public:

  /// Default constructor
  CSCGeometry();

  /// Destructor
  virtual ~CSCGeometry();

  //---- Base class' interface

  // Return a vector of all det types
  virtual const DetTypeContainer&  detTypes() const;

  // Returm a vector of all GeomDetUnit
  virtual const DetUnitContainer& detUnits() const;

  // Returm a vector of all GeomDet (including all GeomDetUnits)
  virtual const DetContainer& dets() const;
  
  // Returm a vector of all GeomDetUnit DetIds
  virtual const DetIdContainer&    detUnitIds() const;

  // Returm a vector of all GeomDet DetIds (including those of GeomDetUnits)
  virtual const DetIdContainer& detIds() const;

  // Return the pointer to the GeomDetUnit corresponding to a given DetId
  virtual const GeomDetUnit* idToDetUnit(DetId) const;

  // Return the pointer to the GeomDet corresponding to a given DetId
  virtual const GeomDet* idToDet(DetId) const;

  //---- Extension of the interface

  /// Return the chamber corresponding to given DetId
  const CSCChamber* chamber(CSCDetId id) const;

  /// Return the orresponding to given DetId
  const CSCLayer* layer(CSCDetId id) const;

  /// Return a vector of all chambers
  const ChamberContainer& chambers() const;

  /// Return a vector of all layers
  const LayerContainer& layers() const;

 private:

  friend class CSCGeometryBuilderFromDDD;

  friend class GeometryAligner;


  /// Add a chamber with given DetId.
  void addChamber(CSCChamber* ch);
  
  /// Add a DetUnit
  void addLayer(CSCLayer* l);

  /// Add a DetType
  void addDetType(GeomDetType* type);

  /// Add a DetId
  void addDetId(DetId id);

  /// Add a GeomDet; not to be called by the builder.
  void addDet(GeomDet* det);

  // The chambers are owned by the geometry (which in turn own layers)
  ChamberContainer  theChambers; 

  // Map for efficient lookup by DetId 
  CSCDetMap         theMap;

  // These are used rarely; they could be computed at runtime 
  // to save memory.
  DetTypeContainer  theDetTypes;
  DetContainer      theDets;       // all dets (chambers and layers)
  DetUnitContainer  theDetUnits;   // all layers
  DetIdContainer    theDetIds;
  DetIdContainer    theDetUnitIds;

  // These are reduntant copies, to satisfy the interface.
  LayerContainer    theLayers;
};

#endif

