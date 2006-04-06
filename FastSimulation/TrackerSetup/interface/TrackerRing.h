#ifndef _TrackerRing_H_
#define _TrackerRing_H_

/** A class that gives some properties of the Tracker Rings in FAMOS
 */

class TrackerRing {
public:
  
  /// Default constructor
  TrackerRing() :
    theInnerRadius(0.),
    theOuterRadius(0.),
    theResolutionAlongX(0.),
    theResolutionAlongY(0.),
    theHitEfficiency(0.)         { }

  /// constructor from private members
  TrackerRing( double theInnerRadius,
	       double theOuterRadius,
	       double theResolutionAlongX = 0.,
	       double theResolutionAlongY = 0.,
	       double theHitEfficiency    = 1. ) :
    theInnerRadius(theInnerRadius),
    theOuterRadius(theOuterRadius),
    theResolutionAlongX(theResolutionAlongX),
    theResolutionAlongY(theResolutionAlongY),
    theHitEfficiency(theHitEfficiency)         { }

  /// Returns the inner radius in cm 
  inline double innerRadius() const { return theInnerRadius; }

  /// Returns the outer radius in cm 
  inline double outerRadius() const { return theOuterRadius; }

  /// Returns the resolution along x in cm (local coordinates)
  inline double resolutionAlongxInCm() const { return theResolutionAlongX; }

  /// Returns the resolution along y in cm(local coordinates)
  inline double resolutionAlongyInCm() const { return theResolutionAlongY; }

  /// Returns the hit reconstruction efficiency
  inline double hitEfficiency() const { return theHitEfficiency; }

private:

  double theInnerRadius;
  double theOuterRadius;
  double theResolutionAlongX;
  double theResolutionAlongY;
  double theHitEfficiency;

};
#endif

