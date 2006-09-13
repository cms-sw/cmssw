#ifndef Geometry_RPCSimAlgo_RPCRoll_H
#define Geometry_RPCSimAlgo_RPCRoll_H

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "Geometry/Vector/interface/GlobalPoint.h"

class StripTopology;
class RPCRollSpecs;
class RPCChamber;
class RPCRoll : public GeomDetUnit{

 public:
  
  RPCRoll(RPCDetId id, BoundPlane::BoundPlanePointer bp, RPCRollSpecs* rrs, const RPCChamber* ch=0);
  ~RPCRoll();
  const RPCRollSpecs* specs() const;
  DetId geographicalId() const;
  RPCDetId id() const;
  const Topology& topology() const;
  const StripTopology& specificTopology() const;
  const GeomDetType& type() const; 
 
  /// Return the chamber this SL belongs to (0 if any, eg if a SL is
  /// built on his own)
    
    const RPCChamber* chamber() const;
  
  int nstrips() const;

  LocalPoint  centreOfStrip(int strip) const;
  LocalPoint  centreOfStrip(float strip) const;
  LocalError  localError(float strip) const;

  float strip(const LocalPoint& lp) const;
  float pitch() const;
  float localPitch(const LocalPoint& lp) const; 
  bool isBarrel() const; 
  bool isForward() const;
  
 private:
  RPCDetId _id;
  RPCRollSpecs* _rrs;
  const RPCChamber* theCh; // NOT owned
};

#endif
