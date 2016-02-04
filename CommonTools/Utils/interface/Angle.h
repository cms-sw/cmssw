#ifndef CommonTools_Utils_Angle_h
#define CommonTools_Utils_Angle_h

/* \class Angle
 *
 * returns three-dimensional Angle between two objects;
 * defined via scalar product: 
 *   angle = acos((v1 * v2)/(|v1| * |v2|))
 *
 * \author Christian Veelken, UC Davis
 */

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/angle.h"

template<typename T1, typename T2 = T1>
struct Angle {
  double operator()( const T1 & t1, const T2 & t2 ) const {
    return angle( t1, t2 );
  }
};

#endif
