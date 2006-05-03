#ifndef Geometry_RPCSimAlgo_RPCRoll_H
#define Geometry_RPCSimAlgo_RPCRoll_H

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "Geometry/Vector/interface/GlobalPoint.h"

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
  
  int nstrips();

  LocalPoint  centreOfStrip(int strip);
  LocalPoint  centreOfStrip(float strip);
  LocalError  localError(float strip);

  float strip(const LocalPoint& lp);
  float pitch();
  float localPitch(const LocalPoint& lp);
  bool isBarrel();
  bool isForward();
  
 private:
  const StripTopology* striptopology();
 private:
  const StripTopology* top_; 
 private:
  RPCDetId _id;
  RPCRollSpecs* _rrs;
};

#endif
