#ifndef Geometry_GEMGeometry_ME0EtaPartition_H
#define Geometry_GEMGeometry_ME0EtaPartition_H

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

class StripTopology;
class ME0EtaPartitionSpecs;
//class ME0Chamber;

class ME0EtaPartition : public GeomDetUnit
{
public:
  
  ME0EtaPartition(ME0DetId id, const BoundPlane::BoundPlanePointer& bp, ME0EtaPartitionSpecs* rrs);
  ~ME0EtaPartition() override;

  const ME0EtaPartitionSpecs* specs() const { return specs_; }
  ME0DetId id() const { return id_; }

  const Topology& topology() const override;
  const StripTopology& specificTopology() const;

  const Topology& padTopology() const;
  const StripTopology& specificPadTopology() const;

  const GeomDetType& type() const override; 
 
  /// Return the chamber this roll belongs to 
  //const ME0Chamber* chamber() const;
 
  // strip-related methods:

  /// number of readout strips in partition
  int nstrips() const;

  /// returns center of strip position for INTEGER strip number
  /// that has a value range of [1, nstrip]
  LocalPoint  centreOfStrip(int strip) const;

  /// returns center of strip position for FRACTIONAL strip number
  /// that has a value range of [0., nstrip]
  LocalPoint  centreOfStrip(float strip) const;
  LocalError  localError(float strip) const;

  /// returns fractional strip number [0..nstrips] for a LocalPoint
  /// E.g., if local point hit strip #2, the fractional strip number would be
  /// somewhere in the (1., 2] interval
  float strip(const LocalPoint& lp) const;

  float pitch() const;
  float localPitch(const LocalPoint& lp) const;
 

  // ME0-CSC pad-related methods:
  
  /// number of ME0-CSC trigger readout pads in partition
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

  ME0DetId id_;
  ME0EtaPartitionSpecs* specs_;
};

#endif

