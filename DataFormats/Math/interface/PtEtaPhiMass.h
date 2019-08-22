#ifndef DataFormats_Math_PtEtaPhiMass_h
#define DataFormats_Math_PtEtaPhiMass_h

#include <cmath>
inline float __attribute__((always_inline)) __attribute__((pure)) etaFromXYZ(float x, float y, float z) {
  float t(z / std::sqrt(x * x + y * y));
  return ::asinhf(t);
}
inline float __attribute__((always_inline)) __attribute__((pure)) etaFromRZ(float r, float z) {
  float t(z / r);
  return ::asinhf(t);
}

/// standard cms four-momentum Lorentz vector
/// consistent with LeafCandidate representation
class PtEtaPhiMass {
private:
  float pt_, eta_, phi_, mass_;

public:
  // default constructor (unitialized)
  PtEtaPhiMass() {}

  //positional constructor (still compatible with Root, c++03)
  constexpr PtEtaPhiMass(float ipt, float ieta, float iphi, float imass)
      : pt_(ipt), eta_(ieta), phi_(iphi), mass_(imass) {}

  /// transverse momentum
  constexpr float pt() const { return pt_; }
  /// momentum pseudorapidity
  constexpr float eta() const { return eta_; }
  /// momentum azimuthal angle
  constexpr float phi() const { return phi_; }
  /// mass
  constexpr float mass() const { return mass_; }
};

class RhoEtaPhi {
private:
  float rho_, eta_, phi_;

public:
  // default constructor (unitialized)
  RhoEtaPhi() {}

  //positional constructor (still compatible with Root, c++03)
  RhoEtaPhi(float irho, float ieta, float iphi) : rho_(irho), eta_(ieta), phi_(iphi) {}

  /// transverse momentum
  float rho() const { return rho_; }
  /// momentum pseudorapidity
  float eta() const { return eta_; }
  /// momentum azimuthal angle
  float phi() const { return phi_; }
};

#endif
