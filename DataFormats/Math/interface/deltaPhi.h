#ifndef DataFormats_Math_deltaPhi_h
#define DataFormats_Math_deltaPhi_h
/* function to compute deltaPhi
 *
 * Ported from original code in RecoJets 
 * by Fedor Ratnikov, FNAL
 */
#include <cmath>

namespace reco {

  inline double deltaPhi(double phi1, double phi2) { 
    double result = phi1 - phi2;
    while (result > M_PI) result -= 2*M_PI;
    while (result <= -M_PI) result += 2*M_PI;
    return result;
  }

  inline double deltaPhi(float phi1, double phi2) {
    return deltaPhi(static_cast<double>(phi1), phi2);
  }
  
  inline double deltaPhi(double phi1, float phi2) {
    return deltaPhi(phi1, static_cast<double>(phi2));
  }
  

  inline float deltaPhi(float phi1, float phi2) { 
    float result = phi1 - phi2;
    while (result > float(M_PI)) result -= float(2*M_PI);
    while (result <= -float(M_PI)) result += float(2*M_PI);
    return result;
  }

  /*
  inline double deltaPhi(float phi1, float phi2) {
    return deltaPhi(static_cast<double>(phi1),
		    static_cast<double>(phi2));
  } 
  */

  template<typename T1, typename T2>
    inline double deltaPhi(T1& t1, T2 & t2) {
    return deltaPhi(t1.phi(), t2.phi());
  }      

  template <typename T> 
    inline T deltaPhi (T phi1, T phi2) { 
    T result = phi1 - phi2;
    while (result > M_PI) result -= 2*M_PI;
    while (result <= -M_PI) result += 2*M_PI;
    return result;
  }

}

// lovely!  VI
using reco::deltaPhi;

template<typename T1, typename T2 = T1>
struct DeltaPhi {
  double operator()(const T1 & t1, const T2 & t2) const {
    return reco::deltaPhi(t1, t2);
  }
};

#endif
