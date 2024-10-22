#include "DataFormats/PatCandidates/interface/ResolutionHelper.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <cmath>
#include <iostream>

using namespace std;  // faster sqrt, sqrt, pow(x,2)....

void pat::helper::ResolutionHelper::rescaleForKinFitter(pat::CandKinResolution::Parametrization parametrization,
                                                        AlgebraicSymMatrix44 &covariance,
                                                        const math::XYZTLorentzVector &initialP4) {
  double inv;
  switch (parametrization) {
    case pat::CandKinResolution::Cart:
    case pat::CandKinResolution::Spher:
      // for us parameter[3] = mass, for KinFitter parameter[3] = mass/initialP4.mass();
      inv = 1.0 / initialP4.mass();
      for (int i = 0; i < 4; i++) {
        covariance(3, i) *= inv;
      }
      covariance(3, 3) *= inv;
      break;
    case pat::CandKinResolution::ESpher:
      // for us parameter[3] = energy, for KinFitter parameter[3] = energy/initialP4.energy();
      inv = 1.0 / initialP4.energy();
      for (int i = 0; i < 4; i++) {
        covariance(3, i) *= inv;
      }
      covariance(3, 3) *= inv;
      break;
    default:;  // nothing to do
  }
}

double pat::helper::ResolutionHelper::getResolP(pat::CandKinResolution::Parametrization parametrization,
                                                const AlgebraicSymMatrix44 &covariance,
                                                const pat::CandKinResolution::LorentzVector &p4) {
  switch (parametrization) {
    // ==== CARTESIAN ====
    case pat::CandKinResolution::Cart:
    case pat::CandKinResolution::ECart:
    case pat::CandKinResolution::MCCart: {
      // p2 = px^2 + py^2 + pz^2
      // dp/dp_i = p_i/p ==> it is a unit vector!
      AlgebraicVector3 derivs(p4.X(), p4.Y(), p4.Z());
      derivs.Unit();
      return sqrt(ROOT::Math::Similarity(derivs, covariance.Sub<AlgebraicSymMatrix33>(0, 0)));
    }
    // ==== SPHERICAL (P)  ====
    case pat::CandKinResolution::Spher:
    case pat::CandKinResolution::ESpher:
    case pat::CandKinResolution::MCSpher:
      return sqrt(covariance(0, 0));
    // ==== SPHERICAL (1/P)  ====
    case pat::CandKinResolution::MCPInvSpher:
      return sqrt(covariance(0, 0)) * (p4.P2());
    // ==== HEP CYLINDRICAL (with Pt = Et, P = E) ====
    case pat::CandKinResolution::EtThetaPhi:
    case pat::CandKinResolution::EtEtaPhi:
      return getResolE(parametrization, covariance, p4);
    // ==== OTHER ====
    case pat::CandKinResolution::Invalid:
      throw cms::Exception("Invalid parametrization") << parametrization;
    default:
      throw cms::Exception("Not Implemented")
          << "getResolP not yet implemented for parametrization " << parametrization;
  }
}
double pat::helper::ResolutionHelper::getResolPt(pat::CandKinResolution::Parametrization parametrization,
                                                 const AlgebraicSymMatrix44 &covariance,
                                                 const pat::CandKinResolution::LorentzVector &p4) {
  switch (parametrization) {
    // ==== CARTESIAN ====
    case pat::CandKinResolution::Cart:
    case pat::CandKinResolution::ECart:
    case pat::CandKinResolution::MCCart: {
      double pti2 = 1.0 / (p4.Perp2());
      return sqrt((covariance(0, 0) * p4.Px() * p4.Px() + covariance(1, 1) * p4.Py() * p4.Py() +
                   2 * covariance(0, 1) * p4.Px() * p4.Py()) *
                  pti2);
    }
    // ==== SPHERICAL (P, Theta)  ====
    case pat::CandKinResolution::Spher:
    case pat::CandKinResolution::ESpher:
    case pat::CandKinResolution::MCSpher: {
      // pt = p * sin(theta)
      double a = sin(p4.Theta());
      double b = p4.P() * cos(p4.Theta());
      return sqrt(a * a * covariance(0, 0) + b * b * covariance(1, 1) + 2 * a * b * covariance(0, 1));
    }
    // ==== SPHERICAL (1/P)  ====
    case pat::CandKinResolution::MCPInvSpher: {
      // pt = (1/pi) * sin(theta)
      double p = p4.P();
      double a = -(p * p) * sin(p4.Theta());
      double b = p * cos(p4.Theta());
      return sqrt(a * a * covariance(0, 0) + b * b * covariance(1, 1) + 2 * a * b * covariance(0, 1));
    }
    // ==== HEP CYLINDRICAL (with Pt = Et) ====
    case pat::CandKinResolution::EtThetaPhi:
    case pat::CandKinResolution::EtEtaPhi:
      return sqrt(covariance(0, 0));
    case pat::CandKinResolution::Invalid:
      throw cms::Exception("Invalid parametrization") << parametrization;
    default:
      throw cms::Exception("Not Implemented")
          << "getResolPt not yet implemented for parametrization " << parametrization;
  }
}

double pat::helper::ResolutionHelper::getResolPInv(pat::CandKinResolution::Parametrization parametrization,
                                                   const AlgebraicSymMatrix44 &covariance,
                                                   const pat::CandKinResolution::LorentzVector &p4) {
  switch (parametrization) {
    // ==== SPHERICAL (P)  ====
    case pat::CandKinResolution::Spher:
    case pat::CandKinResolution::ESpher:
    case pat::CandKinResolution::MCSpher:
      return 1.0 / p4.P2() * sqrt(covariance(0, 0));
    // ==== SPHERICAL (1/P)  ====
    case pat::CandKinResolution::MCPInvSpher:
      return sqrt(covariance(0, 0));
    // ==== OTHER ====
    case pat::CandKinResolution::Cart:
    case pat::CandKinResolution::ECart:
    case pat::CandKinResolution::MCCart:
    case pat::CandKinResolution::EtThetaPhi:
    case pat::CandKinResolution::EtEtaPhi:
      return 1.0 / p4.P2() * getResolP(parametrization, covariance, p4);
    case pat::CandKinResolution::Invalid:
      throw cms::Exception("Invalid parametrization") << parametrization;
    default:
      throw cms::Exception("Not Implemented")
          << "getResolPInv not yet implemented for parametrization " << parametrization;
  }
}

double pat::helper::ResolutionHelper::getResolPx(pat::CandKinResolution::Parametrization parametrization,
                                                 const AlgebraicSymMatrix44 &covariance,
                                                 const pat::CandKinResolution::LorentzVector &p4) {
  switch (parametrization) {
    // ==== CARTESIAN ====
    case pat::CandKinResolution::Cart:
    case pat::CandKinResolution::ECart:
    case pat::CandKinResolution::MCCart:
      return sqrt(covariance(0, 0));
    // ==== SPHERICAL (P)  ====
    case pat::CandKinResolution::Spher:
    case pat::CandKinResolution::ESpher:
    case pat::CandKinResolution::MCSpher:
    case pat::CandKinResolution::MCPInvSpher: {
      // Px = P * sin(theta) * cos(phi)
      double p = p4.P();
      AlgebraicVector3 derivs;
      derivs[0] = sin(p4.Theta()) * cos(p4.Phi());  // now let's hope gcc does common subexpr optimiz.
      if (parametrization == pat::CandKinResolution::MCPInvSpher) {
        derivs[0] *= -(p * p);
      }
      derivs[1] = p * cos(p4.Theta()) * cos(p4.Phi());
      derivs[2] = p * sin(p4.Theta()) * -sin(p4.Phi());
      return sqrt(ROOT::Math::Similarity(derivs, covariance.Sub<AlgebraicSymMatrix33>(0, 0)));
    }
    // ==== HEP CYLINDRICAL (with Pt = Et) ====
    case pat::CandKinResolution::EtThetaPhi:
    case pat::CandKinResolution::EtEtaPhi: {
      // Px = Pt * cos(phi)
      double a = cos(p4.Phi());
      double b = -p4.Pt() * sin(p4.Phi());
      return sqrt(a * a * covariance(0, 0) + 2 * a * b * covariance(2, 0) + b * b * covariance(2, 2));
    }
    // ==== OTHERS ====
    case pat::CandKinResolution::Invalid:
      throw cms::Exception("Invalid parametrization") << parametrization;
    default:
      throw cms::Exception("Not Implemented")
          << "getResolPx not yet implemented for parametrization " << parametrization;
  }
}
double pat::helper::ResolutionHelper::getResolPy(pat::CandKinResolution::Parametrization parametrization,
                                                 const AlgebraicSymMatrix44 &covariance,
                                                 const pat::CandKinResolution::LorentzVector &p4) {
  switch (parametrization) {
    // ==== CARTESIAN ====
    case pat::CandKinResolution::Cart:
    case pat::CandKinResolution::ECart:
    case pat::CandKinResolution::MCCart:
      return sqrt(covariance(1, 1));
    // ==== SPHERICAL (P)  ====
    case pat::CandKinResolution::Spher:
    case pat::CandKinResolution::ESpher:
    case pat::CandKinResolution::MCSpher:
    case pat::CandKinResolution::MCPInvSpher: {
      // Py = P * sin(theta) * sin(phi)
      double p = p4.P();
      AlgebraicVector3 derivs;
      derivs[0] = sin(p4.Theta()) * sin(p4.Phi());  // now let's hope gcc does common subexpr optimiz.
      if (parametrization == pat::CandKinResolution::MCPInvSpher) {
        derivs[0] *= -(p * p);
      }
      derivs[1] = p * cos(p4.Theta()) * sin(p4.Phi());
      derivs[2] = p * sin(p4.Theta()) * cos(p4.Phi());
      return sqrt(ROOT::Math::Similarity(derivs, covariance.Sub<AlgebraicSymMatrix33>(0, 0)));
    }
    // ==== HEP CYLINDRICAL (with Pt = Et) ====
    case pat::CandKinResolution::EtThetaPhi:
    case pat::CandKinResolution::EtEtaPhi: {
      // Py = Pt * sin(phi)
      double a = sin(p4.Phi());
      double b = p4.Pt() * cos(p4.Phi());
      return sqrt(a * a * covariance(0, 0) + 2 * a * b * covariance(2, 0) + b * b * covariance(2, 2));
    }
    // ==== OTHERS ====
    case pat::CandKinResolution::Invalid:
      throw cms::Exception("Invalid parametrization") << parametrization;
    default:
      throw cms::Exception("Not Implemented")
          << "getResolPy not yet implemented for parametrization " << parametrization;
  }
}
double pat::helper::ResolutionHelper::getResolPz(pat::CandKinResolution::Parametrization parametrization,
                                                 const AlgebraicSymMatrix44 &covariance,
                                                 const pat::CandKinResolution::LorentzVector &p4) {
  switch (parametrization) {
    // ==== CARTESIAN ====
    case pat::CandKinResolution::Cart:
    case pat::CandKinResolution::ECart:
    case pat::CandKinResolution::MCCart:
      return sqrt(covariance(2, 2));
    // ==== SPHERICAL (P)  ====
    case pat::CandKinResolution::Spher:
    case pat::CandKinResolution::ESpher:
    case pat::CandKinResolution::MCSpher: {
      // Pz = P * cos(theta)
      double a = cos(p4.Theta());
      double b = -p4.P() * sin(p4.Theta());
      return sqrt(a * a * covariance(0, 0) + 2 * a * b * covariance(1, 0) + b * b * covariance(1, 1));
    }
    case pat::CandKinResolution::MCPInvSpher: {
      // Pz = P * cos(theta)
      double p = p4.P();
      double a = -p * p * cos(p4.Theta());
      double b = -p * sin(p4.Theta());
      return sqrt(a * a * covariance(0, 0) + 2 * a * b * covariance(1, 0) + b * b * covariance(1, 1));
    }
    // ==== HEP CYLINDRICAL (with Pt = Et) ====
    case pat::CandKinResolution::EtThetaPhi: {
      // Pz = Pt * ctg(theta)    d ctg(x) = -1/sin^2(x)
      double s = sin(p4.Theta()), c = cos(p4.Theta());
      double a = c / s;
      double b = -p4.Pt() / (s * s);
      return sqrt(a * a * covariance(0, 0) + 2 * a * b * covariance(1, 0) + b * b * covariance(1, 1));
    }
    case pat::CandKinResolution::EtEtaPhi: {
      // Pz = Pt * sinh(eta)
      double a = sinh(p4.Eta());
      double b = p4.Et() * cosh(p4.Eta());
      return sqrt(a * a * covariance(0, 0) + 2 * a * b * covariance(1, 0) + b * b * covariance(1, 1));
    }
    // ==== OTHERS ====
    case pat::CandKinResolution::Invalid:
      throw cms::Exception("Invalid parametrization") << parametrization;
    default:
      throw cms::Exception("Not Implemented")
          << "getResolPz not yet implemented for parametrization " << parametrization;
  }
}

double pat::helper::ResolutionHelper::getResolE(pat::CandKinResolution::Parametrization parametrization,
                                                const AlgebraicSymMatrix44 &covariance,
                                                const pat::CandKinResolution::LorentzVector &p4) {
  switch (parametrization) {
    // ======= ENERGY BASED ==========
    case pat::CandKinResolution::ECart:
    case pat::CandKinResolution::ESpher:
      return sqrt(covariance(3, 3));
      // ======= ET BASED ==========
    case pat::CandKinResolution::EtThetaPhi: {
      // E = Et/Sin(theta)
      double a = 1.0 / sin(p4.Theta());               // dE/dEt
      double b = -a * a * p4.Et() * cos(p4.Theta());  // dE/dTh
      return sqrt(a * a * covariance(0, 0) + b * b * covariance(1, 1) + 2 * a * b * covariance(0, 1));
    }
    case pat::CandKinResolution::EtEtaPhi: {
      // E = Et/Sin(Theta(eta))
      double th = p4.Theta();
      double a = 1.0 / sin(th);          // dE/dEt
      double b = a * p4.Et() * cos(th);  // dE/dEta: dTh/dEta = - 1.0/sin(theta) = - dE/dEt
      return sqrt(a * a * covariance(0, 0) + b * b * covariance(1, 1) + 2 * a * b * covariance(0, 1));
    }
      // ======= MASS BASED ==========
    case pat::CandKinResolution::Cart: {
      AlgebraicVector4 xoE(p4.X(), p4.Y(), p4.Z(), p4.M());
      xoE *= 1 / p4.E();
      return sqrt(ROOT::Math::Similarity(xoE, covariance));
    }
    case pat::CandKinResolution::MCCart: {
      AlgebraicVector4 xoE(p4.X(), p4.Y(), p4.Z(), 0);
      xoE *= 1 / p4.E();
      return sqrt(ROOT::Math::Similarity(xoE, covariance));
    }
    case pat::CandKinResolution::Spher: {
      // E = sqrt(P^2 + m^2)
      double einv = 1.0 / p4.E();
      double a = p4.P() * einv;  // dE/dP
      double b = p4.M() * einv;  // dE/dm
      return sqrt(a * a * covariance(0, 0) + b * b * covariance(3, 3) + 2 * a * b * covariance(0, 3));
    }
    case pat::CandKinResolution::MCSpher: {
      // E = sqrt(P^2 + m^2); |dE/dP| = |P/E| = P/E
      return p4.P() / p4.E() * sqrt(covariance(0, 0));
    }
    case pat::CandKinResolution::MCPInvSpher:  //
    {
      // E = sqrt(P^2 + m^2); |dE/d(1/P)| = P^2 |dE/dP| = P^3/E
      double p = p4.P();
      return p * p * p / p4.E() * sqrt(covariance(0, 0));
    }
      // ======= OTHER ==========
    case pat::CandKinResolution::Invalid:
      throw cms::Exception("Invalid parametrization") << parametrization;
    default:
      throw cms::Exception("Not Implemented")
          << "getResolE not yet implemented for parametrization " << parametrization;
  }
}

double pat::helper::ResolutionHelper::getResolEt(pat::CandKinResolution::Parametrization parametrization,
                                                 const AlgebraicSymMatrix44 &covariance,
                                                 const pat::CandKinResolution::LorentzVector &p4) {
  switch (parametrization) {
    // ======= ENERGY BASED ==========
    case pat::CandKinResolution::ECart: {
      // Et^2 = E^2 * (Pt^2/P^2)
      double pt2 = p4.Perp2();
      double pz2 = ROOT::Math::Square(p4.Pz()), p2 = pt2 + pz2;
      double e2OverP4 = ROOT::Math::Square(p4.E() / p2);
      AlgebraicVector4 derivs(p4.Px(), p4.Py(), p4.Pz(), p4.E());
      derivs *= (1.0 / p4.Et());
      derivs[0] *= pz2 * e2OverP4;
      derivs[1] *= pz2 * e2OverP4;
      derivs[2] *= -pt2 * e2OverP4;
      derivs[3] *= pt2 / p2;
      return sqrt(ROOT::Math::Similarity(derivs, covariance));
    }
    case pat::CandKinResolution::ESpher: {
      // Et = E * Sin(Theta)
      double st = sin(p4.Theta()), ct = cos(p4.Theta());
      return sqrt(st * st * covariance(3, 3) + ROOT::Math::Square(ct * p4.E()) * covariance(1, 1) +
                  2 * st * ct * p4.E() * covariance(1, 3));
    }
      // ======= ET BASED ==========
    case pat::CandKinResolution::EtThetaPhi:
    case pat::CandKinResolution::EtEtaPhi:
      return sqrt(covariance(0, 0));
      // ======= MASS BASED ==========
    case pat::CandKinResolution::Cart:
    case pat::CandKinResolution::MCCart: {
      // Et^2 = E^2 Sin^2(th) = (p^2 + m^2) * (pt^2) / p^2
      double pt2 = p4.Perp2();
      double p2 = pt2 + ROOT::Math::Square(p4.Pz());
      double e2 = p2 + p4.M2();
      double s2 = pt2 / p2, pi2 = 1.0 / p2;
      double et = sqrt(e2 * s2);
      AlgebraicVector4 derivs(p4.Px(), p4.Py(), p4.Pz(), p4.M());
      derivs *= 1.0 / et;
      derivs[0] *= (s2 + e2 * pi2 * (1.0 - pt2 * pi2));  /// dEt/dPx * Et
      derivs[1] *= (s2 + e2 * pi2 * (1.0 - pt2 * pi2));  /// dEt/dPx * Et
      derivs[2] *= (s2 - e2 * pt2 * pi2 * pi2);          /// dEt/dPx * Et
      if (parametrization == pat::CandKinResolution::Cart) {
        derivs[3] *= s2;
        return sqrt(ROOT::Math::Similarity(derivs, covariance));
      } else {
        derivs[3] = 0;
        return sqrt(ROOT::Math::Similarity(derivs, covariance));  // test if Sub<33> is faster
      }
    }
    case pat::CandKinResolution::Spher: {
      // Et = E sin(theta); dE/dM = M/E, dE/dP = P/E
      double s = sin(p4.Theta()), c = cos(p4.Theta());
      double e = p4.E();
      AlgebraicVector4 derivs(p4.P() / e * s, e * c, 0, p4.M() / e * s);
      return sqrt(ROOT::Math::Similarity(derivs, covariance));
    }
    case pat::CandKinResolution::MCSpher: {
      // Et = E sin(theta); dE/dP = P/E
      double s = sin(p4.Theta()), c = cos(p4.Theta());
      double e = p4.E();
      double a = p4.P() * s / e;
      double b = e * c;
      return sqrt(a * a * covariance(0, 0) + b * b * covariance(1, 1) + 2 * a * b * covariance(0, 1));
    }
    case pat::CandKinResolution::MCPInvSpher: {
      // Et = E sin(theta); dE/dP = P/E -> dE/d(1/P) = - P^2 dE/dP = - P^3 / E
      double s = sin(p4.Theta()), c = cos(p4.Theta());
      double p = p4.P(), e = p4.E();
      double a = (-p * p * p / e) * s;
      double b = e * c;
      return sqrt(a * a * covariance(0, 0) + b * b * covariance(1, 1) + 2 * a * b * covariance(0, 1));
    }
      // ======= OTHER ==========
    case pat::CandKinResolution::Invalid:
      throw cms::Exception("Invalid parametrization") << parametrization;
    default:
      throw cms::Exception("Not Implemented")
          << "getResolEt not yet implemented for parametrization " << parametrization;
  }
}

double pat::helper::ResolutionHelper::getResolM(pat::CandKinResolution::Parametrization parametrization,
                                                const AlgebraicSymMatrix44 &covariance,
                                                const pat::CandKinResolution::LorentzVector &p4) {
  switch (parametrization) {
    // ====== MASS CONSTRAINED =====
    case pat::CandKinResolution::MCSpher:
    case pat::CandKinResolution::MCPInvSpher:
    case pat::CandKinResolution::MCCart:
    case pat::CandKinResolution::EtThetaPhi:
    case pat::CandKinResolution::EtEtaPhi:
      return 0;
    // ======= MASS BASED ==========
    case pat::CandKinResolution::Cart:
    case pat::CandKinResolution::Spher:
      return sqrt(covariance(3, 3));
    // ======= ENERGY BASED ==========
    case pat::CandKinResolution::ESpher: {  // M^2 = E^2 - P^2
      double dMdE = p4.E() / p4.M(), dMdP = -p4.P() / p4.M();
      return sqrt(dMdP * dMdP * covariance(0, 0) + 2 * dMdP * dMdE * covariance(0, 3) + dMdE * dMdE * covariance(3, 3));
    }
    case pat::CandKinResolution::ECart: {  // M^2 = E^2 - sum_i P_i^2
      AlgebraicVector4 derivs(-p4.Px(), -p4.Py(), -p4.Pz(), p4.E());
      derivs *= 1.0 / p4.M();
      return sqrt(ROOT::Math::Similarity(derivs, covariance));
    }
      throw cms::Exception("Not Implemented")
          << "getResolM not yet implemented for parametrization " << parametrization;
    case pat::CandKinResolution::Invalid:
      throw cms::Exception("Invalid parametrization") << parametrization;
    default:
      throw cms::Exception("Not Implemented")
          << "getResolM not yet implemented for parametrization " << parametrization;
  }
}

inline double DetaDtheta(double theta) {
  // y  = -ln(tg(x/2)) =>
  // y' = - 1/tg(x/2) * 1/(cos(x/2))^2 * 1/2 = - 1 / (2 * sin(x/2) * cos(x/2)) = -1/sin(x)
  return -1.0 / sin(theta);
}
inline double DthetaDeta(double eta) {
  // y = 2 atan(exp(-x))
  // y' = 2 * 1/(1+exp^2) * exp(-x) * (-1) = - 2 * exp/(1+exp^2) = - 2 / (exp + 1/exp)
  double e = exp(-eta);
  return -2.0 / (e + 1.0 / e);
}

double pat::helper::ResolutionHelper::getResolEta(pat::CandKinResolution::Parametrization parametrization,
                                                  const AlgebraicSymMatrix44 &covariance,
                                                  const pat::CandKinResolution::LorentzVector &p4) {
  switch (parametrization) {
    case pat::CandKinResolution::Cart:
    case pat::CandKinResolution::ECart:
    case pat::CandKinResolution::MCCart:
      // dEta = dTheta * dEta/dTheta
      return abs(DetaDtheta(p4.Theta())) * getResolTheta(parametrization, covariance, p4);
    case pat::CandKinResolution::ESpher:       //  all the ones which have
    case pat::CandKinResolution::MCPInvSpher:  //  theta as parameter 1
    case pat::CandKinResolution::MCSpher:
    case pat::CandKinResolution::EtThetaPhi:
    case pat::CandKinResolution::Spher:
      return sqrt(covariance(1, 1)) * abs(DetaDtheta(p4.Theta()));
    case pat::CandKinResolution::EtEtaPhi:  // as simple as that
      return sqrt(covariance(1, 1));
    case pat::CandKinResolution::Invalid:
      throw cms::Exception("Invalid parametrization") << parametrization;
    default:
      throw cms::Exception("Not Implemented")
          << "getResolEta not yet implemented for parametrization " << parametrization;
  }
}
double pat::helper::ResolutionHelper::getResolTheta(pat::CandKinResolution::Parametrization parametrization,
                                                    const AlgebraicSymMatrix44 &covariance,
                                                    const pat::CandKinResolution::LorentzVector &p4) {
  switch (parametrization) {
    case pat::CandKinResolution::Cart:
    case pat::CandKinResolution::ECart:
    case pat::CandKinResolution::MCCart: {
      // theta = acos( pz / p )        ; d acos(x) = - 1 / sqrt( 1 - x*x) dx = - p/pt dx
      double pt2 = p4.Perp2();
      double p = p4.P(), pi = 1.0 / p, pi3 = pi * pi * pi;
      double dacos = -p / sqrt(pt2);
      AlgebraicVector3 derivs;
      derivs[0] = -p4.Px() * p4.Pz() * dacos * pi3;
      derivs[1] = -p4.Py() * p4.Pz() * dacos * pi3;
      derivs[2] = pt2 * dacos * pi3;
      return sqrt(ROOT::Math::Similarity(derivs, covariance.Sub<AlgebraicSymMatrix33>(0, 0)));
    }
    case pat::CandKinResolution::ESpher:       //  all the ones which have
    case pat::CandKinResolution::MCPInvSpher:  //  theta as parameter 1
    case pat::CandKinResolution::MCSpher:
    case pat::CandKinResolution::EtThetaPhi:
    case pat::CandKinResolution::Spher:
      return sqrt(covariance(1, 1));
    case pat::CandKinResolution::EtEtaPhi:
      return sqrt(covariance(1, 1)) * abs(DthetaDeta(p4.Eta()));
    case pat::CandKinResolution::Invalid:
      throw cms::Exception("Invalid parametrization") << parametrization;
    default:
      throw cms::Exception("Not Implemented")
          << "getResolTheta not yet implemented for parametrization " << parametrization;
  }
}
double pat::helper::ResolutionHelper::getResolPhi(pat::CandKinResolution::Parametrization parametrization,
                                                  const AlgebraicSymMatrix44 &covariance,
                                                  const pat::CandKinResolution::LorentzVector &p4) {
  double pt2 = p4.Perp2();
  switch (parametrization) {
    case pat::CandKinResolution::Cart:
    case pat::CandKinResolution::ECart:
    case pat::CandKinResolution::MCCart:
      return sqrt(ROOT::Math::Square(p4.Px()) * covariance(1, 1) + ROOT::Math::Square(p4.Py()) * covariance(0, 0) +
                  -2 * p4.Px() * p4.Py() * covariance(0, 1)) /
             pt2;
    case pat::CandKinResolution::ESpher:       //  all the ones which have
    case pat::CandKinResolution::MCPInvSpher:  //  phi as parameter 2
    case pat::CandKinResolution::MCSpher:
    case pat::CandKinResolution::EtThetaPhi:
    case pat::CandKinResolution::Spher:
    case pat::CandKinResolution::EtEtaPhi:
      return sqrt(covariance(2, 2));
    case pat::CandKinResolution::Invalid:
      throw cms::Exception("Invalid parametrization") << parametrization;
    default:
      throw cms::Exception("Not Implemented")
          << "getResolPhi not yet implemented for parametrization " << parametrization;
  }
}
