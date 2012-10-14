#ifndef Ellipse_h
#define Ellipse_h

#include "DataFormats/Math/interface/deltaPhi.h"

namespace reco{
  
  template <typename T1, typename T2>
  inline double ellipse (const T1& t1, const T2& t2, double rPhi, double rEta){
    double dEta = t1.eta()-t2.eta();
    double dPhi = deltaPhi(t1.phi(), t2.phi());
    double distance = (dEta*dEta)/(rEta*rEta) + (dPhi*dPhi)/(rPhi*rPhi);
    return distance;
  }

}
#endif 

