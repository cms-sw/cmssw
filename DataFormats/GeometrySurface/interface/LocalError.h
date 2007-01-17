#ifndef Geom_LocalError_H
#define Geom_LocalError_H

/** A very simple class for 2D error matrix components,
 *  used for the local frame.
 */

#include <cmath>
#include <iosfwd>

class LocalError {
public:

  LocalError() : thexx(0), thexy(0), theyy(0) {}

  LocalError( float xx, float xy, float yy) :
    thexx(xx), thexy(xy), theyy(yy) {}

  float xx() const { return thexx;}
  float xy() const { return thexy;}
  float yy() const { return theyy;}

  /** Return a new LocalError, scaled by a factor interpreted as a 
   *  number of sigmas (standard deviations).
   *  The error matrix components are actually multiplied with the square 
   *  of the factor.
   */
  LocalError scale(float s) { 
    float s2 = s*s;
    return LocalError(s2*xx(), s2*xy(), s2*yy());
  }

  /// Return a new LocalError, rotated by an angle defined by the direction (x,y)
  LocalError rotate(float x, float y) { 
    double mag = sqrt(x*x+y*y);
    return rotateCosSin( x/mag, y/mag);
  }

  /// Return a new LocalError, rotated by an angle phi
  LocalError rotate(float phi) { 
    return rotateCosSin( cos(phi), sin(phi));
  }

  /// Return a new LocalError, rotated by an angle defined by it's cosine and sine
  LocalError rotateCosSin( double c, double s) {
    return LocalError( c*c*xx() - 2*c*s*xy() + s*s*yy(),
		       c*s*xx() + (c*c-s*s)*xy() - c*s*yy(),
		       s*s*xx() + 2*c*s*xy() + c*c*yy());
  }

private:

  float thexx;
  float thexy;
  float theyy;

};  

std::ostream & operator<<( std::ostream& s, const LocalError& err) ;

#endif // LocalError_H
