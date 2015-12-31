#ifndef Math_notmalizedPhi_h
#define Math_notmalizedPhi_h
#include "DataFormats/Math/interface/deltaPhi.h"
/* return a value of phi into interval [-pi,+pi]
 *
 */
template<typename T>
inline
T normalizedPhi(T phi) { return reco::reduceRange(phi);}

#endif
