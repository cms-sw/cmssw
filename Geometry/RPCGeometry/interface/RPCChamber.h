#ifndef RPCGeometry_RPCChamber_h
#define RPCGeometry_RPCChamber_h

/** \class RPCChamber
 *
 *  Model of a RPC chamber.
 *   
 *  A chamber is a GeomDet.
 *  The chamber is composed by 2 or 3 Roll (GeomDetUnit).
 *
 *  $Date: 2011/09/27 09:13:42 $
 *  $Revision: 1.4 $
 *  \author R. Trentadue
 */

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

class RPCRoll;

class RPCChamber : public GeomDet {
public:
  /// Constructor
  RPCChamber(RPCDetId id, const ReferenceCountingPointer<BoundPlane>& plane);
  /// Destructor
  virtual ~RPCChamber();

  /// Return the RPCChamberId of this chamber
  RPCDetId id() const;

  // Which subdetector
  virtual SubDetector subDetector() const {return GeomDetEnumerators::RPCBarrel;}

  /// equal if the id is the same
  bool operator==(const RPCChamber& ch) const;

  /// Add Roll to the chamber which takes ownership
  void add(RPCRoll* rl);

  /// Return the rolls in the chamber
  virtual std::vector< const GeomDet*> components() const;

  /// Return the sub-component (roll) with a given id in this chamber
  virtual const GeomDet* component(DetId id) const;

  /// Return the Roll corresponding to the given id 
  const RPCRoll* roll(RPCDetId id) const;

  const RPCRoll* roll(int isl) const;
  
  /// Return the Rolls
  const std::vector<const RPCRoll*>& rolls() const;

  /// Retunr numbers of rolls
  int nrolls() const;

private:

  RPCDetId theId;

  // The chamber owns its Rolls
  std::vector<const RPCRoll*> theRolls;

};
#endif
