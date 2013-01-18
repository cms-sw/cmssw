#ifndef Geometry_GEMGeometry_GEMEtaPartition_H
#define Geometry_GEMGeometry_GEMEtaPartition_H

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

class StripTopology;
class GEMEtaPartitionSpecs;
//class GEMChamber;

class GEMEtaPartition : public GeomDetUnit
{
public:
  
  GEMEtaPartition(GEMDetId id, BoundPlane::BoundPlanePointer bp, GEMEtaPartitionSpecs* rrs);
  ~GEMEtaPartition();

  const GEMEtaPartitionSpecs* specs() const;
  GEMDetId id() const;

  const Topology& topology() const;
  const StripTopology& specificTopology() const;

  const Topology& padTopology() const;
  const StripTopology& specificPadTopology() const;

  const GeomDetType& type() const; 
 
  /// Return the chamber this roll belongs to 
  //const GEMChamber* chamber() const;
 
  // strip-related methods:

  int nstrips() const;

  LocalPoint  centreOfStrip(int strip) const;
  LocalPoint  centreOfStrip(float strip) const;
  LocalError  localError(float strip) const;

  // [0..nstrips)
  float strip(const LocalPoint& lp) const;
  float pitch() const;
  float localPitch(const LocalPoint& lp) const;
 

  // pad-related methods:
  
  int npads() const;

  LocalPoint  centreOfPad(int pad) const;
  LocalPoint  centreOfPad(float pad) const;

  // [0..npads)
  float pad(const LocalPoint& lp) const;
  float padPitch() const;
  float localPadPitch(const LocalPoint& lp) const;


  // relations between strips and pads:
  
  // [0..npads)
  float padOfStrip(int strip) const;
  int firstStripInPad(int pad) const;
  int lastStripInPad(int pad) const;

private:

  //  void setChamber(const GEMChamber* ch);

  GEMDetId _id;
  GEMEtaPartitionSpecs* _rrs;
  //  const GEMChamber* theCh; // NOT owned
};

#endif
