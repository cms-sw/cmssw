/** \file
 *
 * $Date:  22/02/2006 16:01:31 CET $
 * $Revision: 1.0 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 */

/* This Class Header */
#include "DataFormats/DTRecHit/interface/DTRecSegment2D.h"

/* Collaborating Class Header */

/* C++ Headers */
#include <iostream>
using namespace std;

/* ====================================================================== */

/// Constructor
DTRecSegment2D::DTRecSegment2D() {
}

/// Destructor
DTRecSegment2D::~DTRecSegment2D() {
}

/* Operations */ 
LocalError DTRecSegment2D::localPositionError() const {
  return LocalError(theCovMatrix[1][1],0.,0.);
}

LocalError DTRecSegment2D::localDirectionError() const{
  return LocalError(theCovMatrix[0][0],0.,0.);
}

int DTRecSegment2D::degreesOfFreedom() const {
  //TODO
  return 0;
}

ostream& operator<<(ostream& os, const DTRecSegment2D& seg) {
  os << "Pos " << seg.localPosition() << 
    " Dir: " << seg.localDirection() <<
    " chi2/ndof: " << seg.chi2() << "/" << seg.degreesOfFreedom() ;
  return os;
}
