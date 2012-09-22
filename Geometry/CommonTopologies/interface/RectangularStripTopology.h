#ifndef Geometry_CommonTopologies_RectangularStripTopology_H
#define Geometry_CommonTopologies_RectangularStripTopology_H

/** Specialised strip topology for rectangular barrel detectors.
 *  The strips are parallel to the local Y axis, so X is the precisely
 *  measured coordinate.
 */

#include "Geometry/CommonTopologies/interface/StripTopology.h"

class RectangularStripTopology GCC11_FINAL : public StripTopology {
public:

  RectangularStripTopology(int nstrips, float pitch, float detlength);

  virtual LocalPoint localPosition(float strip) const;

  virtual LocalPoint localPosition(const MeasurementPoint&) const;
  
  virtual LocalError 
  localError(float strip, float stripErr2) const;
  
  virtual LocalError 
  localError(const MeasurementPoint&, const MeasurementError&) const;
  
  virtual float strip(const LocalPoint&) const;

  // the number of strip span by the segment between the two points..
  virtual float coveredStrips(const LocalPoint& lp1, const LocalPoint& lp2)  const ; 


  virtual MeasurementPoint measurementPosition(const LocalPoint&) const;
    
  virtual MeasurementError 
  measurementError(const LocalPoint&, const LocalError&) const;

  virtual int channel(const LocalPoint& lp) const {  return std::min(int(strip(lp)),theNumberOfStrips-1); }

  virtual float pitch() const { return thePitch; }

  virtual float localPitch(const LocalPoint&) const { return thePitch;}
  
  virtual float stripAngle(float strip) const {  return 0;}

  virtual int nstrips() const { return theNumberOfStrips;}

  virtual float stripLength() const {return theStripLength;}

  virtual float localStripLength(const LocalPoint& /*aLP*/) const {
    return stripLength();
  }

private:
  float thePitch;
  int   theNumberOfStrips;
  float theStripLength;
  float theOffset;   
};

#endif


