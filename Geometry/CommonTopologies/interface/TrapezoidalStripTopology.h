#ifndef Geometry_CommonTopologies_TrapezoidalStripTopology_H
#define Geometry_CommonTopologies_TrapezoidalStripTopology_H

/** Specialised strip topology for rectangular barrel detectors.
 *  The strips are parallel to the local Y axis, so X is the precisely
 *  measured coordinate.
 */

#include "Geometry/CommonTopologies/interface/StripTopology.h"


/** Specialization of StripTopology for detectors of symmetric trapezoidal
 *  shape. The local Y coordinate is parallel to the central strip,
 *  and prpendicular to the paralle sides of the trapezoid.
 *  The first and last strips are parallel to the corresponding sides
 *  of the trapezoid.
 *  The pitch is constant at constant y.
 *  This topology makes a non-linear transformation: the pitch is 
 *  not constant along the strips.
 */

class TrapezoidalStripTopology final : public StripTopology {
public:

  /** constructed from:
   *    number of strips
   *    pitch in the middle 
   *    det heigth (strip length in the middle)
   *    radius of circle passing through the middle of the det
   *    with center at the crossing of the two sides.
   */
  TrapezoidalStripTopology(int nstrip, float pitch, float detheight,float r0);

  TrapezoidalStripTopology(int nstrip, float pitch, float detheight,float r0, int yAx);

  using StripTopology::localPosition;
  LocalPoint localPosition(float strip) const override;

  LocalPoint localPosition(const MeasurementPoint&) const override;
  
  using StripTopology::localError;
  LocalError 
  localError(float strip, float stripErr2) const override;
  
  LocalError 
  localError(const MeasurementPoint&, const MeasurementError&) const override;
  
  float strip(const LocalPoint&) const override;

  MeasurementPoint measurementPosition(const LocalPoint&) const override;
    
  MeasurementError 
  measurementError(const LocalPoint&, const LocalError&) const override;

  int channel(const LocalPoint&) const override;

  /** Pitch in the middle of the DetUnit */
  float pitch() const override; 

  float localPitch(const LocalPoint&) const override;
  
  /** angle between strip and symmetry axis */
  float stripAngle(float strip) const override;

  int nstrips() const override; 

  /// det heigth (strip length in the middle)
  float stripLength() const override {return theDetHeight;}
  float localStripLength(const LocalPoint& aLP) const override;
  
  /** radius of circle passing through the middle of the det
   *    with center at the crossing of the two sides.
   */
  float radius() const { return theDistToBeam;}

protected:

  virtual float shiftOffset( float pitch_fraction);

private:
  int   theNumberOfStrips;
  float thePitch;    // pitch at the middle of the det. plane
  float theOffset;  
  float theDistToBeam; 
  float theDetHeight; 
  int   theYAxOr;
};


#endif


