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

typedef std::map<DetId, Pointer2Chamber> MapId2Chamber;
typedef std::vector<CSCChamber*> ChamberContainer;
typedef std::vector<CSCLayer*> LayerContainer;

class CSCGeometry : public TrackingGeometry {
 public:

  /// Default constructor
  CSCGeometry();

  /// Destructor
  virtual ~CSCGeometry();

  /// Return a vector of all det types
  virtual const DetTypeContainer&  detTypes() const;

  /// Returm a vector of all GeomDetUnit
  virtual const DetContainer& dets() const;
  
  /// Returm a vector of all DetIds
  virtual const DetIdContainer& detIds() const;

  /// Return the pointer to the GeomDetUnit corresponding to a given DetId
  virtual const GeomDetUnit* idToDet(DetId) const;

  /// Add a DetUnit
  void addDet(GeomDetUnit* p);

  /// Add a DetType
  void addDetType( GeomDetType* );

  /// Add a DetId
  void addDetId(DetId p);

  /// Add a chamber with given DetId
  void addChamber( CSCDetId id, Pointer2Chamber chamber);

  /// Get the chamber corresponding to given DetId
  Pointer2Chamber getChamber( CSCDetId ) const;

  /// Vector of chambers
    const ChamberContainer chambers() const;

  /// Vector of layers
    const LayerContainer layers() const;

 private:
  
  DetTypeContainer  theDetTypes;
  DetContainer      theDets;
  DetIdContainer    theDetIds;
  mapIdToDet        theMap;

  MapId2Chamber theSystemOfChambers; //@@ FIXME when GeomDet composite ready

};

#endif

