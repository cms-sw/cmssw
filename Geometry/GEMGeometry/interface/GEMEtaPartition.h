#ifndef Geometry_GEMGeometry_GEMEtaPartition_H
#define Geometry_GEMGeometry_GEMEtaPartition_H

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

class StripTopology;
class GEMEtaPartitionSpecs;

class GEMEtaPartition : public GeomDetUnit
{
public:
  
  GEMEtaPartition(GEMDetId id, BoundPlane::BoundPlanePointer bp, GEMEtaPartitionSpecs* rrs);
  ~GEMEtaPartition();

  const GEMEtaPartitionSpecs* specs() const { return specs_; }
  GEMDetId id() const { return id_; }

  const Topology& topology() const;
  const StripTopology& specificTopology() const;

  const Topology& padTopology() const;
  const StripTopology& specificPadTopology() const;

  const GeomDetType& type() const; 
 
  // strip-related methods:

  /// number of readout strips in partition
  int nstrips() const;

  /// returns center of strip position for INTEGER strip number
  /// that has a value range of [1, nstrip]
  LocalPoint  centreOfStrip(int strip) const;

  /// returns center of strip position for FRACTIONAL strip number
  /// that has a value range of [0., nstrip]
  LocalPoint  centreOfStrip(float strip) const;
  LocalError  localError(float strip, float cluster_size = 1.) const;

  /// returns fractional strip number [0..nstrips] for a LocalPoint
  /// E.g., if local point hit strip #2, the fractional strip number would be
  /// somewhere in the (1., 2] interval
  float strip(const LocalPoint& lp) const;

  float pitch() const;
  float localPitch(const LocalPoint& lp) const;
 

  // GEM-CSC pad-related methods:
  
  /// number of GEM-CSC trigger readout pads in partition
  int npads() const;

  /// returns center of pad position for INTEGER pad number
  /// that has a value range of [1, npads]
  LocalPoint  centreOfPad(int pad) const;

  /// returns center of pad position for FRACTIONAL pad number
  /// that has a value range of [0., npads]
  LocalPoint  centreOfPad(float pad) const;

  /// returns FRACTIONAL pad number [0.,npads] for a point
  float pad(const LocalPoint& lp) const;

  /// pad pitch in a center
  float padPitch() const;
  /// pad pitch at a particular point
  float localPadPitch(const LocalPoint& lp) const;


  // relations between strips and pads:
  
  /// returns FRACTIONAL pad number [0.,npads] for an integer strip [1,nstrip]
  float padOfStrip(int strip) const;

  /// returns first strip (INT number [1,nstrip]) for pad (an integer [1,npads])
  int firstStripInPad(int pad) const;

  /// returns last strip (INT number [1,nstrip]) for pad (an integer [1,npads])
  int lastStripInPad(int pad) const;

private:

  GEMDetId id_;
  GEMEtaPartitionSpecs* specs_;
};

#endif

