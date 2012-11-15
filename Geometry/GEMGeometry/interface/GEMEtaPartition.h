#ifndef Geometry_GEMGeometry_GEMEtaPartition_H
#define Geometry_GEMGeometry_GEMEtaPartition_H

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

class StripTopology;
class GEMEtaPartitionSpecs;
//class GEMChamber;
class GEMEtaPartition : public GeomDetUnit{

 public:
  
  GEMEtaPartition(GEMDetId id, BoundPlane::BoundPlanePointer bp, GEMEtaPartitionSpecs* rrs);
  ~GEMEtaPartition();
  const GEMEtaPartitionSpecs* specs() const;
  GEMDetId id() const;
  const Topology& topology() const;
  const StripTopology& specificTopology() const;
  const GeomDetType& type() const; 
 
  /// Return the chamber this roll belongs to 
  //const GEMChamber* chamber() const;
  
  int nstrips() const;

  LocalPoint  centreOfStrip(int strip) const;
  LocalPoint  centreOfStrip(float strip) const;
  LocalError  localError(float strip) const;

  float strip(const LocalPoint& lp) const;
  float pitch() const;
  float localPitch(const LocalPoint& lp) const; 
  
 private:
  //  void setChamber(const GEMChamber* ch);

 private:
  GEMDetId _id;
  GEMEtaPartitionSpecs* _rrs;
  //  const GEMChamber* theCh; // NOT owned
};

#endif
