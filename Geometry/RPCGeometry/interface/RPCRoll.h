#ifndef Geometry_RPCSimAlgo_RPCRoll_H
#define Geometry_RPCSimAlgo_RPCRoll_H

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

class StripTopology;
class RPCRollSpecs;
class RPCRoll : public GeomDetUnit{

 public:
  
  RPCRoll( BoundPlane* bp, RPCRollSpecs* rrs, RPCDetId id);
  ~RPCRoll();
  const RPCRollSpecs* specs() const;
  DetId geographicalId() const;
  RPCDetId id() const;
  const Topology& topology() const;
  const GeomDetType& type() const; 
  
 public:
  
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
  const StripTopology* striptopology() const;
 private:
  mutable const StripTopology* top_; 
 private:
  RPCDetId _id;
  RPCRollSpecs* _rrs;
};

#endif
