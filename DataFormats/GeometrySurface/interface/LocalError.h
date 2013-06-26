#ifndef Geom_LocalError_H
#define Geom_LocalError_H

/** A very simple class for 2D error matrix components,
 *  used for the local frame.
 */

#include "DataFormats/GeometrySurface/interface/TrivialError.h"
#include <cmath>
#include <iosfwd>

class LocalError {
public:
  LocalError() : thexx(0), thexy(0), theyy(0) {}
  LocalError(InvalidError) : thexx(-9999.e10f), thexy(0), theyy(-9999.e10f) {}

  LocalError( float xx, float xy, float yy) :
    thexx(xx), thexy(xy), theyy(yy) {}

  bool invalid() const { return thexx<-1.e10f;}
  bool valid() const { return !invalid();}


  float xx() const { return thexx;}
  float xy() const { return thexy;}
  float yy() const { return theyy;}

  /** Return a new LocalError, scaled by a factor interpreted as a 
   *  number of sigmas (standard deviations).
   *  The error matrix components are actually multiplied with the square 
   *  of the factor.
   */
  LocalError scale(float s) const { 
    float s2 = s*s;
    return LocalError(s2*xx(), s2*xy(), s2*yy());
  }

  /// Return a new LocalError, rotated by an angle defined by the direction (x,y)
  LocalError rotate(float x, float y) const { 
    return rotateCosSin( x, y, 1.f/(x*x+y*y) );
  }

  /// Return a new LocalError, rotated by an angle phi
  LocalError rotate(float phi) const { 
    return rotateCosSin( cos(phi), sin(phi));
  }

  /// Return a new LocalError, rotated by an angle defined by it's cosine and sine
  LocalError rotateCosSin( float c, float s, float mag2i=1.f) const {
    return LocalError( mag2i*( (c*c)*xx() + (s*s)*yy() - 2.f*(c*s)*xy()),
		       mag2i*( (c*s)*(xx() - yy()) +  (c*c-s*s)*xy()) ,
		       mag2i*( (s*s)*xx() + (c*c)*yy() + 2.f*(c*s)*xy())
                     );
  }

private:

  float thexx;
  float thexy;
  float theyy;

};  

std::ostream & operator<<( std::ostream& s, const LocalError& err) ;

#endif // LocalError_H
