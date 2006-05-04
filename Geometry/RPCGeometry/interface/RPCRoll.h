#ifndef Geometry_RPCGeometry_RPCRoll_H
#define Geometry_RPCGeometry_RPCRoll_H

/** \class RPCRoll
 *
 * Describes the lowest level detector unit.
 *
 * \author M. Maggi - INFN Bari
 */

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "Geometry/RPCGeometry/interface/RPCRollSpecs.h"

class RPCRoll : public GeomDetUnit{

 public:
  RPCRoll( BoundPlane* bp, RPCRollSpecs* rrs, RPCDetId id) :
    GeomDetUnit(bp), _id(id),_rrs(rrs){}
  ~RPCRoll(){}
  const RPCRollSpecs* specs() const {return _rrs;}
  DetId geographicalId() const {return _id;}
  RPCDetId id() const {return _id;}
  const Topology& topology() const {return _rrs->topology();}
  const GeomDetType& type() const {return (*_rrs);}

 private:
  RPCDetId _id;
  RPCRollSpecs* _rrs;
};

#endif
