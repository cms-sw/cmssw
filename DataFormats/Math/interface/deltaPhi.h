#ifndef DataFormats_Math_deltaPhi_h
#define DataFormats_Math_deltaPhi_h
/* function to compute deltaPhi
 *
 * Ported from original code in RecoJets 
 * by Fedor Ratnikov, FNAL
 *
 * Protection against junk input + protection from protection in FW Lite
 * by Petar Maksimovic, JHU
 */
#include <cmath>

// The following header files are needed to turn on LogError (or exceptions)
// and report crap that was supplied by the caller.  (It happens when there
// is a data corruption upstream.)  In that case the code will return the 
// simple difference and yank the LogError.  The #if !def CINT, etc. is
// needed to still make this function usable within ROOT/FW Lite macros.
// 
#if !defined(__CINT__) && !defined(__MAKECINT__)
// For exceptions, use #include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#else
#include <iostream>
#include "TMath.h"
static const double M_PI = TMath::Pi();   /* Needed when running in CINT... */
#endif /* !defined(__CINT__) ... */

namespace reco {

  // Protection if phi1 or phi2 are junk: avoid an almost-infinite loop...
  static const double MAXPILOOPS = 314159.3;   // 100,000*M_PI
  inline void errorInputsTooLarge() {
    char errstr[] = "Invalid input: either phi1 or phi2 are way too large.";
#if !defined(__CINT__) && !defined(__MAKECINT__) 
    // For exceptions, use: throw cms::Exception("DeltaPhi") 
    edm::LogError("DeltaPhi") << "[Math/deltaPhi::]" << errstr;
#else
    // This branch is executed only in CINT.
    std::cout << "DeltaPhi::error: " << errstr << std::endl;
#endif
  }


  inline double deltaPhi(double phi1, double phi2) { 
    double result = phi1 - phi2;
    if (fabs(result) > MAXPILOOPS) {
      errorInputsTooLarge();
      return result;
    }
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

 template <typename T> 
   T deltaPhi (T phi1, T phi2) { 
   T result = phi1 - phi2;
   if (fabs(result) > MAXPILOOPS) {
     errorInputsTooLarge();
     return result;
   }
   while (result > M_PI) result -= 2*M_PI;
   while (result <= -M_PI) result += 2*M_PI;
   return result;
 }

}

using reco::deltaPhi;

template<typename T1, typename T2 = T1>
struct DeltaPhi {
  double operator()(const T1 & t1, const T2 & t2) const {
    return reco::deltaPhi(t1, t2);
  }
};

#endif
