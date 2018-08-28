#ifndef Geometry_GEMGeometry_GEMEtaPartition_H
#define Geometry_GEMGeometry_GEMEtaPartition_H

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

class StripTopology;
class GEMEtaPartitionSpecs;

class GEMEtaPartition : public GeomDet
{
public:
  
  GEMEtaPartition(GEMDetId id, const BoundPlane::BoundPlanePointer& bp, GEMEtaPartitionSpecs* rrs);
  ~GEMEtaPartition() override;

  const GEMEtaPartitionSpecs* specs() const { return specs_; }
  GEMDetId id() const { return id_; }

  const Topology& topology() const override;
  const StripTopology& specificTopology() const;

  const Topology& padTopology() const;
  const StripTopology& specificPadTopology() const;

  const GeomDetType& type() const override; 
 
  // strip-related methods:

  /// number of readout strips in partition
  int nstrips() const;

  /// returns center of strip position for INTEGER strip number
  /// that has a value range of [0, nstrip-1]
  LocalPoint  centreOfStrip(int strip) const;

  /// returns center of strip position for FRACTIONAL strip number
  /// that has a value range of [0.0, nstrip)
  LocalPoint  centreOfStrip(float strip) const;
  LocalError  localError(float strip, float cluster_size= 1.) const;
  /// returns fractional strip number [0.0, nstrips) for a LocalPoint
  /// E.g., if local point hit strip #2, the fractional strip number would be
  /// somewhere in the [2.0, 3.0) interval
  float strip(const LocalPoint& lp) const;

  float pitch() const;
  float localPitch(const LocalPoint& lp) const;
 

  // GEM-CSC pad-related methods:
  
  /// number of GEM-CSC trigger readout pads in partition
  int npads() const;

  /// returns center of pad position for INTEGER pad number
  /// that has a value range of [0, npads-1]
  LocalPoint  centreOfPad(int pad) const;

  /// returns center of pad position for FRACTIONAL pad number
  /// that has a value range of [0., npads)
  LocalPoint  centreOfPad(float pad) const;

  /// returns FRACTIONAL pad number [0.,npads) for a point
  float pad(const LocalPoint& lp) const;

  /// pad pitch in a center
  float padPitch() const;
  /// pad pitch at a particular point
  float localPadPitch(const LocalPoint& lp) const;


  // relations between strips and pads:
  
  /// returns FRACTIONAL pad number [0.,npads) for an integer strip [0,nstrip-1]
  float padOfStrip(int strip) const;

  /// returns first strip (INT number [0,nstrip-1]) for pad (an integer [0,npads-1])
  int firstStripInPad(int pad) const;

  /// returns last strip (INT number [0,nstrip-1]) for pad (an integer [0,npads-1])
  int lastStripInPad(int pad) const;

private:

  GEMDetId id_;
  GEMEtaPartitionSpecs* specs_;
};

#endif

