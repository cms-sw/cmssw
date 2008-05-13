#ifndef PhysicsTools_Utilities_deltaPhi_h
#define PhysicsTools_Utilities_deltaPhi_h
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
  
  inline double deltaPhi(float phi1, float phi2) {
    return deltaPhi(static_cast<double>(phi1),
		    static_cast<double>(phi2));
 } 

  template<typename T1, typename T2>
    inline double deltaPhi(const T1& t1, const T2 & t2) {
    return deltaPhi(t1.phi(), t2.phi());
  }      

}

#endif
