#ifndef _GEM_STRIP_TOPOLOGY_H_
#define _GEM_STRIP_TOPOLOGY_H_

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

  int nstrips() const override { return theNumberOfStrips; }

  float stripLength() const override { return theDetHeight; }

  float localStripLength(const LocalPoint&) const override;

  float angularWidth() const { return theAngularWidth; }
  float detHeight() const { return theDetHeight; }
  float yExtentOfStripPlane() const { return theDetHeight; }
  float centreToIntersection() const { return theCentreToIntersection; }
  float radius() const { return theCentreToIntersection; }
  float originToIntersection() const { return (theCentreToIntersection - yCentre); }

  float yDistanceToIntersection(float y) const;
  float xOfStrip(int strip, float y) const;
  float yAxisOrientation() const { return theYAxisOrientation; }
  float phiOfOneEdge() const { return thePhiOfOneEdge; }
  float yCentreOfStripPlane() const { return yCentre; }

private:
  int theNumberOfStrips;          // total no. of strips in plane of strips
  float theAngularWidth;          // angle subtended by each strip = phi pitch
  float theDetHeight;             // length of long symmetry axis = twice the apothem of the enclosing trapezoid
  float theCentreToIntersection;  // distance centre of detector face to intersection of edge strips (projected)
  float thePhiOfOneEdge;          // local 'phi' of one edge of plane of strips (I choose it negative!)
  float theYAxisOrientation;      // 1 means y axis going from smaller to larger side, -1 means opposite direction
  float yCentre;  // Non-zero if offset in local y between midpoint of detector (strip plane) extent and local origin.
};

#endif
