#ifndef RPCGeometry_RPCGeometry_h
#define RPCGeometry_RPCGeometry_h

/** \class RPCGeometry
 *
 *  The model of the geometry of RPC.
 *
 *  \author M. Maggi - INFN Bari
 */

#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include <vector>
#include <map>

class GeomDetType;
class GeomDetUnit;

class RPCGeometry : public TrackingGeometry {
  typedef std::map<DetId, RPCRoll*> RPCRollMap;

 public:
  /// Default constructor
  RPCGeometry();

  /// Destructor
  virtual ~RPCGeometry();

  // Return a vector of all det types
  virtual const DetTypeContainer&  detTypes() const;

  // Return a vector of all GeomDetUnit
  virtual const DetUnitContainer& detUnits() const;

  // Return a vector of all GeomDet
  virtual const DetContainer& dets() const;
  
  // Return a vector of all GeomDetUnit DetIds
  virtual const DetIdContainer& detUnitIds() const;

  // Return a vector of all GeomDet DetIds
  virtual const DetIdContainer& detIds() const;

  // Return the pointer to the GeomDetUnit corresponding to a given DetId
  virtual const GeomDetUnit* idToDetUnit(DetId) const;

  // Return the pointer to the GeomDet corresponding to a given DetId
  virtual const GeomDet* idToDet(DetId) const;


  //---- Extension of the interface

  // Return a vector of all RPC chambers
  //const std::vector<RPCChamber*>& chambers() const;

  /// Return a vector of all RPC rolls
  const std::vector<RPCRoll*>& rolls() const;

  // Return a RPCChamber given its id
  //const RPCChamber* chamber(RPCDetId id) const;

  /// Return a roll given its id
  const RPCRoll* roll(RPCDetId id) const;

  /// Add a RPC roll to the Geometry
  void add(RPCRoll* roll);

 private:
  DetUnitContainer theRolls;
  DetContainer theDets;
  DetTypeContainer theRollTypes;
  DetIdContainer theRollIds;
  mapIdToDet     idtoDet;
  mapIdToDetUnit idtoRoll;
  std::vector<RPCRoll*> allRolls;

};

#endif
