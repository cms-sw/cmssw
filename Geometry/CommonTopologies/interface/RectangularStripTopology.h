#ifndef Geometry_CommonTopologies_RectangularStripTopology_H
#define Geometry_CommonTopologies_RectangularStripTopology_H

/** Specialised strip topology for rectangular barrel detectors.
 *  The strips are parallel to the local Y axis, so X is the precisely
 *  measured coordinate.
 */

#include "Geometry/CommonTopologies/interface/StripTopology.h"

class RectangularStripTopology final : public StripTopology {
public:
  RectangularStripTopology(int nstrips, float pitch, float detlength);

  using StripTopology::localPosition;
  LocalPoint localPosition(float strip) const override;

  LocalPoint localPosition(const MeasurementPoint&) const override;

  using StripTopology::localError;
  LocalError localError(float strip, float stripErr2) const override;

  LocalError localError(const MeasurementPoint&, const MeasurementError&) const override;

  float strip(const LocalPoint&) const override;

  // the number of strip span by the segment between the two points..
  float coveredStrips(const LocalPoint& lp1, const LocalPoint& lp2) const override;

  MeasurementPoint measurementPosition(const LocalPoint&) const override;

  MeasurementError measurementError(const LocalPoint&, const LocalError&) const override;

  int channel(const LocalPoint& lp) const override { return std::min(int(strip(lp)), theNumberOfStrips - 1); }

  float pitch() const override { return thePitch; }

  float localPitch(const LocalPoint&) const override { return thePitch; }

  float stripAngle(float strip) const override { return 0; }

  int nstrips() const override { return theNumberOfStrips; }

  float stripLength() const override { return theStripLength; }

  float localStripLength(const LocalPoint& /*aLP*/) const override { return stripLength(); }

private:
  float thePitch;
  int theNumberOfStrips;
  float theStripLength;
  float theOffset;
};

#endif
