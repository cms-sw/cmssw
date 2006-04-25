#ifndef Alignment_CommonAlignment_AlignableDeltaPhi_H
#define Alignment_CommonAlignment_AlignableDeltaPhi_H

#include <CLHEP/Units/PhysicalConstants.h>

/** returns the smalles difference in phi (radians) either either
 *  clockwise or anticlocwise, whatever is smaller
 */

float AlignableDeltaPhi( const float a, const float b) {
  float deltaPhi;
  deltaPhi = abs(a - b );
  //two points cannot be further away then pi
  if (deltaPhi > pi * rad ) {
    deltaPhi = 2.*pi - deltaPhi;
  }
  return deltaPhi;
}

#endif // AlignableDeltaPhi_H
