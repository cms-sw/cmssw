#ifndef MuonIsolation_Direction_H
#define MuonIsolation_Direction_H 

/** \class muonisolation::Direction
 *  Simple eta-phi direction
 *  \author M. Zanetti
 */

#include <cmath>
#include <sstream>
#include <iostream>

namespace muonisolation {

class  Direction {
public:

  struct Distance { 
    float deltaR; 
    float relativeAngle; 
    bool operator < (const Distance & rd2) const { return deltaR < rd2.deltaR; }; 
  };
  
  Direction(double eta = 0., double phi = 0.) : theEta(eta), thePhi(phi) {
    while( thePhi < 0.0 ) thePhi += 2*M_PI; 
    while( thePhi >= 2*M_PI ) thePhi -= 2*M_PI;
  };

  double eta() const { return theEta; }
  double phi() const { return thePhi; } 
  double theta() const { return acos(tanh(theEta)); } 

  inline bool operator==(const Direction & d2) {
    if ( this == &d2 ) return true;
    if ( deltaR(d2) < 1.e-4) return true;
    return false; 
  }

  inline double deltaR2(const Direction & dir2) const {
    double dEta = theEta - dir2.eta();
    double dPhi = fabs( thePhi - dir2.phi());
    while (dPhi > M_PI) dPhi -= 2*M_PI;
    return dEta*dEta + dPhi*dPhi;
  }
  inline double deltaR(const Direction & dir2) const {
    return std::sqrt( deltaR2(dir2) );
  }

  inline double deltaR2(double eta, double phi) const {
    double dEta = theEta - eta;
    double dPhi = fabs( thePhi - phi);
    while (dPhi > M_PI) dPhi -= 2*M_PI;
    return dEta*dEta + dPhi*dPhi;
  }
  inline double deltaR(double eta, double phi) const {
    return std::sqrt(deltaR2(eta,phi));
  }


  Distance operator- (const Direction & dir2) const {
    Distance result;
    double dR    = deltaR(dir2);
    double dEta = theEta-dir2.eta();
    double dPhi = thePhi-dir2.phi();
    while( dPhi < -M_PI ) dPhi += 2*M_PI;  
    while( dPhi >= M_PI ) dPhi -= 2*M_PI;
    result.relativeAngle = (dR > 1.e-4) ? atan2(dPhi,dEta) : 0.;
    result.deltaR = dR;
    return result;
  }

  Direction operator+ (const Distance & relDir) const {
    double eta = theEta + relDir.deltaR*cos(relDir.relativeAngle);
    double phi = thePhi + relDir.deltaR*sin(relDir.relativeAngle);
    return Direction(eta,phi);
  }

 std::string print() const {
   std::ostringstream str;
   str<<" (Eta="<<theEta<< "," << "Phi=" <<thePhi<<")";
   return str.str();
 }

private:
  float  theEta;
  float  thePhi;
};

}

#endif
