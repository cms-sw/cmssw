#ifndef Geometry_RPCSimAlgo_RPCRoll_H
#define Geometry_RPCSimAlgo_RPCRoll_H

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

class StripTopology;
class RPCRollSpecs;
class RPCChamber;
class RPCRoll : public GeomDetUnit{

 public:
  
  RPCRoll(RPCDetId id, BoundPlane::BoundPlanePointer bp, RPCRollSpecs* rrs);
  ~RPCRoll();
  const RPCRollSpecs* specs() const;
  RPCDetId id() const;
  const Topology& topology() const;
  const StripTopology& specificTopology() const;
  const GeomDetType& type() const; 
 
  /// Return the chamber this roll belongs to 
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
  void setChamber(const RPCChamber* ch);

 private:
  RPCDetId _id;
  RPCRollSpecs* _rrs;
  const RPCChamber* theCh; // NOT owned
};

#endif
