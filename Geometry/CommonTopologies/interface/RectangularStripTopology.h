#ifndef Geometry_CommonTopologies_RectangularStripTopology_H
#define Geometry_CommonTopologies_RectangularStripTopology_H

/** Specialised strip topology for rectangular barrel detectors.
 *  The strips are parallel to the local Y axis, so X is the precisely
 *  measured coordinate.
 */

#include "Geometry/CommonTopologies/interface/StripTopology.h"

class RectangularStripTopology : public StripTopology {
public:

  RectangularStripTopology(int nstrips, float pitch, float detlength);

  virtual LocalPoint localPosition(float strip) const;

  virtual LocalPoint localPosition(const MeasurementPoint&) const;
  
  virtual LocalError 
  localError(float strip, float stripErr2) const;
  
  virtual LocalError 
  localError(const MeasurementPoint&, const MeasurementError&) const;
  
  virtual float strip(const LocalPoint&) const;

  virtual MeasurementPoint measurementPosition(const LocalPoint&) const;
    
  virtual MeasurementError 
  measurementError(const LocalPoint&, const LocalError&) const;

  virtual int channel(const LocalPoint&) const;

  virtual float pitch() const; 

  virtual float localPitch(const LocalPoint&) const;
  
  virtual float stripAngle(float strip) const;

  virtual int nstrips() const; 

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


