#ifndef MuonIsolation_Direction_H
#define MuonIsolation_Direction_H 

/** \class L3MuonIsolation::Direction
 *  Simple eta-phi direction
 *  \author M. Zanetti
 */

#include <cmath>
#include <iostream>

namespace muonisolation {

class  Direction {
public:
  
  Direction(float eta = 0., float phi = 0.) {
    eta_ = eta;
    while( phi < 0.0 ) 
      phi += 2*M_PI; 
    while( phi >= 2*M_PI ) 
      phi -= 2*M_PI;
    phi_ = phi ;
  };

  float eta() const { return eta_; }
  float phi() const { return phi_; } 
  float theta() const { return acos(tanh(eta_)); } 

  inline bool operator==(const muonisolation::Direction & d2) {
    if ( this == &d2 ) return true;
    if ( deltaR(d2) < 1.e-4) return true;
    return false; 
  }

  inline float deltaR(const muonisolation::Direction & dir2) const {
    float dEta = eta() - dir2.eta();
    float dPhi = fabs(phi() - dir2.phi());
    while (dPhi > M_PI) dPhi -= 2*M_PI;
    return sqrt( dEta*dEta + dPhi*dPhi);
  }

private:
  float  eta_;
  float  phi_;
};

}

//std::ostream  &operator << (std::ostream &out, const muonisolation::Direction & dir) 
//{ out << "(Eta ="<<dir.eta() << "," << "Phi = " << dir.phi()<<")"; return out; }

#endif
