#ifndef Geometry_CommonTopologies_GEMStripTopology_H
#define Geometry_CommonTopologies_GEMStripTopology_H

/** \class GEMStripTopology
 *  based on CSCRadialStripTopology and TrapezoidalStripTopology
 *  \author Hyunyong Kim - TAMU
 */

#include "Geometry/CommonTopologies/interface/StripTopology.h"

class GEMStripTopology final : public StripTopology {
public:
  GEMStripTopology(int ns, float aw, float dh, float r0);
  GEMStripTopology(int ns, float aw, float dh, float r0, float yAx);
  ~GEMStripTopology() override {}

  using StripTopology::localPosition;
  LocalPoint localPosition(float strip) const override;

  LocalPoint localPosition(const MeasurementPoint&) const override;

  using StripTopology::localError;
  LocalError localError(float strip, float stripErr2) const override;
  LocalError localError(const MeasurementPoint&, const MeasurementError&) const override;

  float strip(const LocalPoint&) const override;

  int nearestStrip(const LocalPoint&) const;

  MeasurementPoint measurementPosition(const LocalPoint&) const override;

  MeasurementError measurementError(const LocalPoint&, const LocalError&) const override;

  int channel(const LocalPoint&) const override;

  float phiPitch(void) const { return angularWidth(); }

  float pitch() const override;

  float localPitch(const LocalPoint&) const override;

  float stripAngle(float strip) const override;

  int nstrips() const override { return numberOfStrips_; }

  float stripLength() const override { return detHeight_; }

  float localStripLength(const LocalPoint&) const override;

  float angularWidth() const { return angularWidth_; }
  float detHeight() const { return detHeight_; }
  float yExtentOfStripPlane() const { return detHeight_; }
  float centreToIntersection() const { return centreToIntersection_; }
  float radius() const { return centreToIntersection_; }
  float originToIntersection() const { return (centreToIntersection_ - yCentre_); }

  float yDistanceToIntersection(float y) const;
  float xOfStrip(int strip, float y) const;
  float yAxisOrientation() const { return yAxisOrientation_; }
  float phiOfOneEdge() const { return phiOfOneEdge_; }
  float yCentreOfStripPlane() const { return yCentre_; }

private:
  int numberOfStrips_;          // total no. of strips in plane of strips
  float angularWidth_;          // angle subtended by each strip = phi pitch
  float detHeight_;             // length of long symmetry axis = twice the apothem of the enclosing trapezoid
  float centreToIntersection_;  // distance centre of detector face to intersection of edge strips (projected)
  float phiOfOneEdge_;          // local 'phi' of one edge of plane of strips (I choose it negative!)
  float yAxisOrientation_;      // 1 means y axis going from smaller to larger side, -1 means opposite direction
  float yCentre_;  // Non-zero if offset in local y between midpoint of detector (strip plane) extent and local origin.
};

#endif
