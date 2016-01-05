#ifndef Math_notmalizedPhi_h
#define Math_notmalizedPhi_h
#include "DataFormats/Math/interface/deltaPhi.h"

// return a value of phi into interval [-pi,+pi]
template<typename T>
inline
T normalizedPhi(T phi) { return reco::reduceRange(phi);}

// cernlib V306
template<typename T>
inline 
T proxim(T b, T a) {
        constexpr T c1 = 2.*M_PI;
        constexpr T c2 = 1/c1;
        return b+c1*std::round(c2*(a-b));
}

template<typename T>
inline
bool checkPhiInRange(T phi, T phi1, T phi2) {
    constexpr T c1 = 2.*M_PI;
    phi1 = normalizedPhi(phi1);
    phi2 = proxim(phi2,phi1);
    // phi & phi1 are in [-pi,pi] range...
    return ( (phi1 <= phi) && (phi <= phi2) ) ||
           ( (phi1 <= phi+c1) && (phi+c1 <= phi2) );
}

#endif
