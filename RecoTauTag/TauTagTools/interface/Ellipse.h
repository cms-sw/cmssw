#ifndef Ellipse_h
#define Ellipse_h

#include "DataFormats/Math/interface/deltaPhi.h"

namespace reco{
  
  inline double deltaEta2(double eta1, double eta2){
    double deltaEta = eta1 - eta2;
    return deltaEta*deltaEta;
  }
  
  template <typename T1, typename T2>
  inline double deltaEta2(const T1& t1, const T2& t2){
    return deltaEta2(t1.eta(), t2.eta());
  }
  
  template <typename T1, typename T2>
  inline double ellipse (const T1& t1, const T2& t2, double rPhi, double rEta){
    double dEta2 = deltaEta2(t1.eta(), t2.eta());
    double dPhi = deltaPhi(t1.phi(), t2.phi());
    double distance = dEta2/(rEta*rEta) + (dPhi*dPhi)/(rPhi*rPhi);
    return distance;
  }

}
#endif 

