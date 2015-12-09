#ifndef Math_angle_h
#define Math_angle_h
/* function to compute 3D angle 
 *
 * Ported from original code in RecoJets 
 * by Fedor Ratnikov, FNAL
 */
#include <cmath>

template <class T>
T angle (T x1, T y1, T z1, T x2, T y2, T z2) {
  return std::acos((x1*x2 + y1*y2 + z1*z2)/std::sqrt((x1*x1 + y1*y1 + z1*z1)*(x2*x2 + y2*y2 + z2*z2)));
}

template<typename T1, typename T2>
double angle( const T1 & t1, const T2 & t2 ) {
  return angle( t1.x(), t1.y(), t1.z(), t2.x(), t2.y(), t2.z() );
} 

#endif
