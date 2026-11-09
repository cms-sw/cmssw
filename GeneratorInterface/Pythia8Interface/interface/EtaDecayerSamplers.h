#ifndef GeneratorInterface_Pythia8Interface_EtaDecayerSamplers_h
#define GeneratorInterface_Pythia8Interface_EtaDecayerSamplers_h

// Unweighted samplers of EtaDecayer (EtaDecayer.h). Each one draws the decay variables from a proposal density, accepts
// with probability weight/bound (accept-reject), builds the daughters in the parent rest frame, orients the decay
// isotropically and boosts to the parent momentum. The dynamics (form factors, amplitudes) enter through functors
// that return the model factor already divided by its bound, i.e. in [0, 1] (see EtaDecayerModels.h). A weight
// outside [0, 1] means a bound is too small; it throws instead of silently biasing the sample.
// flat() returns a uniform random number in (0, 1) from Pythia's generator, which CMSSW seeds.
//
// Decay densities (angular structure as in Petri, arXiv:1010.2378):
//   P -> gamma l+ l-              Kroll-Wada
//   P -> l+ l- l'+ l'- (l != l')  Petri Eq. (3.36)
//   P -> l+ l- l+ l-              direct - exchange interference of the helicity amplitudes (below)
//   P -> pi+ pi- l+ l-            magnetic term, Petri Sec. 3.5.2, Eq. (3.85)
//   eta' -> eta pi pi             Dalitz-plot polynomial (BESIII), isotropic (spin 0)

#include "FWCore/Utilities/interface/Exception.h"
#include "GeneratorInterface/Pythia8Interface/interface/EtaDecayerModels.h"

#include "TLorentzVector.h"
#include "TVector3.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <utility>

namespace gen {
  namespace etadecayer {

    // Maximum number of proposals per decay before giving up.
    constexpr unsigned kMaxAttempts = 1000000;

    inline void checkWeight(double w, const char* sampler) {
      if (!std::isfinite(w) || w < 0 || w > 1)
        throw cms::Exception("EtaDecayer")
            << sampler << ": accept-reject weight " << w << " outside [0, 1], the rejection bound is too small";
    }

    // Isotropic orientation of the decay (polar angle theta and azimuth about z), then the boost to the parent.
    template <std::size_t N, class Flat>
    void orientAndBoost(std::array<TLorentzVector, N>& out, const TLorentzVector& parent, Flat& flat) {
      const double pi = std::acos(-1.);
      const double theta = std::acos(2 * flat() - 1), orientPhi = 2 * pi * flat();
      for (auto& p : out) {
        p.RotateY(theta);
        p.RotateZ(orientPhi);
        p.Boost(parent.BoostVector());
      }
    }

    // P -> l- l+ (isotropic two-body decay). Output: l-, l+.
    template <class Flat>
    std::array<TLorentzVector, 2> twoBody(const TLorentzVector& parent, double m, Flat flat) {
      const double M = parent.M();
      const double pMag = std::sqrt(std::max(0., M * M / 4 - m * m));
      const double c = 2 * flat() - 1, s = std::sqrt(1 - c * c), pi = std::acos(-1.), phi = 2 * pi * flat();
      std::array<TLorentzVector, 2> out = {
          {TLorentzVector(pMag * s * std::cos(phi), pMag * s * std::sin(phi), pMag * c, M / 2),
           TLorentzVector(-pMag * s * std::cos(phi), -pMag * s * std::sin(phi), -pMag * c, M / 2)}};
      for (auto& p : out)
        p.Boost(parent.BoostVector());
      return out;
    }

    // P -> gamma l+ l- (Kroll-Wada). With x = q^2/M^2 and theta the l- angle in the dilepton rest frame,
    //   dGamma/(dx dcos(theta)) ~ (1/x) beta (1-x)^3 (1 + cos^2(theta) + (1 - beta^2) sin^2(theta)) |F(x M^2)|^2.
    // x is proposed log-uniformly (cancels the 1/x photon pole). ffSq(q2) = |F(q2)/F(0)|^2 / bound.
    // Output: l-, l+, gamma.
    template <class FormFactorSq, class Flat>
    std::array<TLorentzVector, 3> singleDalitz(const TLorentzVector& parent, double m, FormFactorSq ffSq, Flat flat) {
      const double M = parent.M(), M2 = M * M;
      const double low = 4 * m * m / M2, high = 1.;
      const double logRange = std::log(high / low), pi = std::acos(-1.);
      for (unsigned attempt = 0; attempt < kMaxAttempts; ++attempt) {
        const double x = low * std::exp(logRange * flat());
        const double beta2 = 1 - 4 * m * m / (x * M2);
        if (!(beta2 > 0))
          continue;
        const double beta = std::sqrt(beta2);
        const double c = 2 * flat() - 1;
        const double weight = beta * std::pow(1 - x, 3) * (1 + c * c + (1 - beta2) * (1 - c * c)) / 2 * ffSq(x * M2);
        checkWeight(weight, "P -> gamma l+ l-");
        if (flat() >= weight)
          continue;
        const double q = M * std::sqrt(x);
        const double k = M * (1 - x) / 2;  // photon momentum in the parent frame
        const double pl = q * beta / 2;
        const double s = std::sqrt(1 - c * c), phi = 2 * pi * flat();
        TLorentzVector gamma(0, 0, k, k);
        TLorentzVector lMinus(pl * s * std::cos(phi), pl * s * std::sin(phi), pl * c, q / 2);
        TLorentzVector lPlus(-pl * s * std::cos(phi), -pl * s * std::sin(phi), -pl * c, q / 2);
        lMinus.Boost(0, 0, -k / (M - k));
        lPlus.Boost(0, 0, -k / (M - k));
        std::array<TLorentzVector, 3> out = {{lMinus, lPlus, gamma}};
        orientAndBoost(out, parent, flat);
        return out;
      }
      throw cms::Exception("EtaDecayer") << "P -> gamma l+ l-: rejection limit exhausted";
    }

    // Angular factor of P -> gamma* gamma* -> (l1- l1+)(l2- l2+), Petri Eq. (3.36), divided by its maximum 2.
    // b1, b2 are the pair velocities squared, c1, c2 the l- polar angles in the pair rest frames, phi the angle
    // between the two decay planes.
    inline double doubleDalitzAngular(double b1, double b2, double c1, double c2, double phi) {
      const double a = b1 * (1 - c1 * c1), b = b2 * (1 - c2 * c2);
      return 1 - (a + b) / 2 + a * b * std::pow(std::sin(phi), 2) / 2;
    }

    // P -> l1+ l1- l2+ l2- without identical leptons:
    //   dGamma/(ds1 ds2 dcos1 dcos2 dphi) ~ lambda^{3/2}(1, s1/M^2, s2/M^2) beta1 beta2 angular |F(s1, s2)|^2 / (s1 s2),
    // with s1, s2 proposed log-uniformly. ff(s1, s2) = |F(s1, s2)/F(0, 0)|^2 / bound. Output: l1-, l1+, l2-, l2+.
    template <class FormFactor, class Flat>
    std::array<TLorentzVector, 4> doubleDalitz(
        const TLorentzVector& parent, double m1, double m2, FormFactor ff, Flat flat) {
      const double M = parent.M(), M2 = M * M;
      const double low1 = 4 * m1 * m1, low2 = 4 * m2 * m2;
      const double log1 = std::log(std::pow(M - 2 * m2, 2) / low1);
      const double log2 = std::log(std::pow(M - 2 * m1, 2) / low2);
      const double pi = std::acos(-1.);
      for (unsigned attempt = 0; attempt < kMaxAttempts; ++attempt) {
        const double s1 = low1 * std::exp(log1 * flat()), s2 = low2 * std::exp(log2 * flat());
        if (std::sqrt(s1) + std::sqrt(s2) >= M)
          continue;
        const double b1 = 1 - low1 / s1, b2 = 1 - low2 / s2;
        const double lambda = kallen(s1 / M2, s2 / M2);
        const double c1 = 2 * flat() - 1, c2 = 2 * flat() - 1, phi = 2 * pi * flat();
        const double weight =
            std::pow(lambda, 1.5) * std::sqrt(b1 * b2) * doubleDalitzAngular(b1, b2, c1, c2, phi) * ff(s1, s2);
        checkWeight(weight, "P -> l+ l- l'+ l'-");
        if (flat() >= weight)
          continue;
        const double k = M * std::sqrt(lambda) / 2;
        const double e1 = (M2 + s1 - s2) / (2 * M), e2 = (M2 + s2 - s1) / (2 * M);
        const double p1 = std::sqrt(s1 * b1) / 2, p2 = std::sqrt(s2 * b2) / 2;
        const double t1 = std::sqrt(1 - c1 * c1), t2 = std::sqrt(1 - c2 * c2);
        // Azimuth of the first plane about the pair axis: the third orientation angle of the decay.
        const double az = 2 * pi * flat();
        std::array<TLorentzVector, 4> out = {
            {TLorentzVector(p1 * t1 * std::cos(az), p1 * t1 * std::sin(az), p1 * c1, std::sqrt(s1) / 2),
             TLorentzVector(-p1 * t1 * std::cos(az), -p1 * t1 * std::sin(az), -p1 * c1, std::sqrt(s1) / 2),
             TLorentzVector(p2 * t2 * std::cos(az + phi), p2 * t2 * std::sin(az + phi), p2 * c2, std::sqrt(s2) / 2),
             TLorentzVector(
                 -p2 * t2 * std::cos(az + phi), -p2 * t2 * std::sin(az + phi), -p2 * c2, std::sqrt(s2) / 2)}};
        for (unsigned i = 0; i < 4; ++i)
          out[i].Boost(0, 0, i < 2 ? k / e1 : -k / e2);
        orientAndBoost(out, parent, flat);
        return out;
      }
      throw cms::Exception("EtaDecayer") << "P -> l+ l- l'+ l'-: rejection limit exhausted";
    }

    // Helicity amplitudes of P -> gamma* gamma* -> l- l+ l- l+ for the direct (D: pairs 01, 23) and exchange
    // (X: pairs 03, 21) diagrams, with pointlike vertices (F = 1). The physical amplitude is D F_D - X F_X.
    namespace fourlepton {
      using Spinor = std::array<Complex, 4>;
      using Current = std::array<Complex, 3>;

      // Dirac-basis spinors with ubar u = 2m and vbar v = -2m; both spin states are summed.
      inline Spinor spinor(const TLorentzVector& p, double m, unsigned spin, bool antiparticle) {
        const double root = std::sqrt(p.E() + m);
        Spinor out{};
        const Complex a = spin == 0 ? Complex(p.Pz(), 0) : Complex(p.Px(), -p.Py());
        const Complex b = spin == 0 ? Complex(p.Px(), p.Py()) : Complex(-p.Pz(), 0);
        if (antiparticle) {
          out[0] = a / root;
          out[1] = b / root;
          out[2 + spin] = root;
        } else {
          out[spin] = root;
          out[2] = a / root;
          out[3] = b / root;
        }
        return out;
      }

      // Spatial components of the vector current ubar gamma^i v.
      inline Current current(const Spinor& u, const Spinor& v) {
        const Complex a = std::conj(u[0]), b = std::conj(u[1]);
        const Complex c = std::conj(u[2]), d = std::conj(u[3]), I(0, 1);
        return {{a * v[3] + b * v[2] + c * v[1] + d * v[0],
                 I * (-a * v[3] + b * v[2] - c * v[1] + d * v[0]),
                 a * v[2] - b * v[3] + c * v[0] - d * v[1]}};
      }

      // Epsilon-tensor contraction of the two currents in the parent rest frame: q . (a x b).
      inline Complex contraction(const Current& a, const Current& b, const TVector3& q) {
        return q.X() * (a[1] * b[2] - a[2] * b[1]) + q.Y() * (a[2] * b[0] - a[0] * b[2]) +
               q.Z() * (a[0] * b[1] - a[1] * b[0]);
      }

      // Momenta p: l-(0), l+(1), l-(2), l+(3). The overall coupling cancels in the sampler.
      class Amplitudes {
      public:
        Amplitudes(std::array<TLorentzVector, 4> p, double m) {
          TLorentzVector parent;
          for (const auto& daughter : p)
            parent += daughter;
          const double M = parent.M();
          for (auto& daughter : p)
            daughter.Boost(-parent.BoostVector());
          const auto qD = p[0] + p[1], rD = p[2] + p[3];
          const auto qX = p[0] + p[3], rX = p[2] + p[1];
          dScale_ = M / (qD.M2() * rD.M2());  // photon propagators
          xScale_ = M / (qX.M2() * rX.M2());
          dAxis_ = qD.Vect();
          xAxis_ = qX.Vect();
          std::array<std::array<Spinor, 2>, 4> spinors;
          for (unsigned i = 0; i < 4; ++i)
            for (unsigned s = 0; s < 2; ++s)
              spinors[i][s] = spinor(p[i], m, s, i % 2 == 1);
          for (unsigned s = 0; s < 2; ++s)
            for (unsigned t = 0; t < 2; ++t) {
              d1_[s][t] = current(spinors[0][s], spinors[1][t]);
              d2_[s][t] = current(spinors[2][s], spinors[3][t]);
              x1_[s][t] = current(spinors[0][s], spinors[3][t]);
              x2_[s][t] = current(spinors[2][s], spinors[1][t]);
            }
        }
        // {D, X} for the lepton spins (a, b, c, d) of particles (0, 1, 2, 3).
        std::array<Complex, 2> operator()(unsigned a, unsigned b, unsigned c, unsigned d) const {
          return {{dScale_ * contraction(d1_[a][b], d2_[c][d], dAxis_),
                   xScale_ * contraction(x1_[a][d], x2_[c][b], xAxis_)}};
        }

      private:
        using Currents = std::array<std::array<Current, 2>, 2>;
        Currents d1_, d2_, x1_, x2_;
        TVector3 dAxis_, xAxis_;
        double dScale_, xScale_;
      };
    }  // namespace fourlepton

    // P -> l+ l- l+ l- (identical leptons), with the form factor applied to each diagram at its own pair masses
    // before the two diagrams interfere. Proposal: the pointlike non-identical density of doubleDalitz, with the two
    // l+ swapped with probability 1/2 (an equal mixture of the direct and exchange pairings). With F_D, F_X the double
    // form factors of the two pairings and B >= max |F|^2, the weight
    //   |D F_D - X F_X|^2 / (2 B (|D|^2 + |X|^2)),
    // summed over the spins, is <= 1 because |a - b|^2 <= 2 (|a|^2 + |b|^2).
    // pairAmp(s1, s2) = F(s1, s2)/F(0, 0) (not divided by the bound). Output: l-, l+, l-, l+.
    template <class PairAmp, class Flat>
    std::array<TLorentzVector, 4> identicalDoubleDalitz(
        const TLorentzVector& parent, double m, PairAmp pairAmp, double bound, Flat flat) {
      const TLorentzVector rest(0, 0, 0, parent.M());
      for (unsigned attempt = 0; attempt < kMaxAttempts; ++attempt) {
        // proposal: one accepted pointlike non-identical decay (its own accept-reject loop)
        auto out = doubleDalitz(rest, m, m, [](double, double) { return 1.; }, flat);
        if (flat() < 0.5)
          std::swap(out[1], out[3]);
        const fourlepton::Amplitudes a(out, m);
        TLorentzVector P;
        for (const auto& d : out)
          P += d;
        std::array<TLorentzVector, 4> r = out;
        for (auto& d : r)
          d.Boost(-P.BoostVector());
        const Complex fD = pairAmp((r[0] + r[1]).M2(), (r[2] + r[3]).M2());
        const Complex fX = pairAmp((r[0] + r[3]).M2(), (r[2] + r[1]).M2());
        double coherent = 0, sum1 = 0;
        for (unsigned i = 0; i < 2; ++i)
          for (unsigned j = 0; j < 2; ++j)
            for (unsigned k = 0; k < 2; ++k)
              for (unsigned l = 0; l < 2; ++l) {
                const auto v = a(i, j, k, l);
                coherent += std::norm(v[0] * fD - v[1] * fX);
                sum1 += std::norm(v[0]) + std::norm(v[1]);
              }
        const double denominator = 2 * bound * sum1;
        if (!(denominator > 0) || !std::isfinite(denominator))
          throw cms::Exception("EtaDecayer") << "P -> l+ l- l+ l-: invalid proposal density";
        const double weight = coherent / denominator;
        // |D F_D - X F_X|^2 <= 2 B (|D|^2 + |X|^2) holds exactly; allow for rounding at the equality
        checkWeight(weight > 1 && weight <= 1 + 1e-12 ? 1. : weight, "P -> l+ l- l+ l-");
        if (flat() >= std::min(1., weight))
          continue;
        for (auto& p : out)
          p.Boost(parent.BoostVector());
        return out;
      }
      throw cms::Exception("EtaDecayer") << "P -> l+ l- l+ l-: rejection limit exhausted";
    }

    // Angular factor of the magnetic P -> pi+ pi- l+ l- amplitude (Petri Eq. (3.85)): cosPi, cosL are the pi+ and
    // l- polar angles in their pair rest frames, phi the angle between the pi pi and l l planes.
    inline double pipiMagneticAngular(double betaL2, double cosPi, double cosL, double phi) {
      return (1 - cosPi * cosPi) * (1 - betaL2 * (1 - cosL * cosL) * std::pow(std::sin(phi), 2));
    }

    // P -> pi+ pi- l+ l-:
    //   dGamma ~ lambda^{3/2}(1, s/M^2, sl/M^2) (s/M^2) beta_pi^3 beta_l angular |f(s, sl)|^2 / sl,
    // s = m_pipi^2 proposed uniformly, sl = m_ll^2 log-uniformly. massWeight(s, sl) = |f(s, sl)|^2 / bound.
    // Output: l-, l+, pi+, pi-.
    template <class MassWeight, class Flat>
    std::array<TLorentzVector, 4> llpipi(
        const TLorentzVector& parent, double ml, double mpi, MassWeight massWeight, Flat flat) {
      const double M = parent.M(), M2 = M * M, smin = 4 * mpi * mpi, qmin = 4 * ml * ml;
      const double smax = std::pow(M - 2 * ml, 2), qmax = std::pow(M - 2 * mpi, 2);
      const double logq = std::log(qmax / qmin), pi = std::acos(-1.);
      // product of the monotonic bounds of the phase-space factors
      const double phaseBound = std::pow(kallen(smin / M2, qmin / M2), 1.5) * (smax / M2) *
                                std::pow(1 - smin / smax, 1.5) * std::sqrt(1 - qmin / qmax);
      for (unsigned attempt = 0; attempt < 10 * kMaxAttempts; ++attempt) {
        const double s = smin + (smax - smin) * flat(), q = qmin * std::exp(logq * flat());
        if (std::sqrt(s) + std::sqrt(q) >= M)
          continue;
        const double bp = 1 - smin / s, bl = 1 - qmin / q, lam = kallen(s / M2, q / M2);
        const double cp = 2 * flat() - 1, cl = 2 * flat() - 1, phi = 2 * pi * flat();
        const double weight = std::pow(lam, 1.5) * (s / M2) * std::pow(bp, 1.5) * std::sqrt(bl) / phaseBound *
                              pipiMagneticAngular(bl, cp, cl, phi) * massWeight(s, q);
        checkWeight(weight, "P -> pi+ pi- l+ l-");
        if (flat() >= weight)
          continue;
        const double k = M * std::sqrt(lam) / 2;
        const double el = (M2 + q - s) / (2 * M), ep = (M2 + s - q) / (2 * M);
        const double pl = std::sqrt(q * bl) / 2, pp = std::sqrt(s * bp) / 2;
        const double tl = std::sqrt(1 - cl * cl), tp = std::sqrt(1 - cp * cp), az = 2 * pi * flat();
        std::array<TLorentzVector, 4> out = {
            {TLorentzVector(pl * tl * std::cos(az + phi), pl * tl * std::sin(az + phi), pl * cl, std::sqrt(q) / 2),
             TLorentzVector(-pl * tl * std::cos(az + phi), -pl * tl * std::sin(az + phi), -pl * cl, std::sqrt(q) / 2),
             TLorentzVector(pp * tp * std::cos(az), pp * tp * std::sin(az), pp * cp, std::sqrt(s) / 2),
             TLorentzVector(-pp * tp * std::cos(az), -pp * tp * std::sin(az), -pp * cp, std::sqrt(s) / 2)}};
        for (unsigned i = 0; i < 4; ++i)
          out[i].Boost(0, 0, i < 2 ? k / el : -k / ep);
        orientAndBoost(out, parent, flat);
        return out;
      }
      throw cms::Exception("EtaDecayer") << "P -> pi+ pi- l+ l-: rejection limit exhausted";
    }

    // P -> l+ l- A with a massive boson A: x = q^2/M^2 proposed log-uniformly between threshold and (M - m_A)^2/M^2,
    // l- polar angle c = cos(theta) in the dilepton rest frame relative to the dilepton direction. weight(x, c) is the
    // density times x, divided by its bound. Output: l-, l+, A.
    template <class Weight, class Flat>
    std::array<TLorentzVector, 3> llA(const TLorentzVector& parent, double mLep, double mA, Weight weight, Flat flat) {
      const double M = parent.M(), M2 = M * M;
      const double xlo = 4 * mLep * mLep / M2, xhi = (M - mA) * (M - mA) / M2;
      const double logRange = std::log(xhi / xlo), pi = std::acos(-1.);
      for (unsigned attempt = 0; attempt < kMaxAttempts; ++attempt) {
        const double x = xlo * std::exp(logRange * flat());
        const double beta2 = 1 - 4 * mLep * mLep / (x * M2);
        if (!(beta2 > 0))
          continue;
        const double beta = std::sqrt(beta2);
        const double c = 2 * flat() - 1;
        const double w = weight(x, c);
        checkWeight(w, "P -> l+ l- A");
        if (flat() >= w)
          continue;
        const double q = M * std::sqrt(x);
        const double p = splitMomentum(M, q, mA);
        if (!(p > 0))
          continue;  // numerically at the kinematic edge
        const double eA = std::sqrt(p * p + mA * mA);
        const double betaZ = -p / (M - eA);
        const double pl = q * beta / 2;
        const double s = std::sqrt(1 - c * c), phi = 2 * pi * flat();
        TLorentzVector a(0, 0, p, eA);
        TLorentzVector lMinus(pl * s * std::cos(phi), pl * s * std::sin(phi), pl * c, q / 2);
        TLorentzVector lPlus(-pl * s * std::cos(phi), -pl * s * std::sin(phi), -pl * c, q / 2);
        lMinus.Boost(0, 0, betaZ);
        lPlus.Boost(0, 0, betaZ);
        std::array<TLorentzVector, 3> out = {{lMinus, lPlus, a}};
        orientAndBoost(out, parent, flat);
        return out;
      }
      throw cms::Exception("EtaDecayer") << "P -> l+ l- A: rejection limit exhausted";
    }

    // P -> pi+ pi- A, A on shell, built as pi+ pi- recoiling against A with the pi+ polar angle cp in the pi pi rest
    // frame. Output: pi+, pi-, A.
    template <class Flat>
    std::array<TLorentzVector, 3> pipiAKinematics(
        const TLorentzVector& parent, double M, double s, double cp, double p, double mA, double mPi, Flat& flat) {
      const double pi = std::acos(-1.);
      const double eA = std::sqrt(p * p + mA * mA), ePP = M - eA;
      const double bp = 1 - 4 * mPi * mPi / s, pp = std::sqrt(s * bp) / 2;
      const double tp = std::sqrt(1 - cp * cp), az = 2 * pi * flat();
      TLorentzVector a(0, 0, p, eA);
      TLorentzVector piPlus(pp * tp * std::cos(az), pp * tp * std::sin(az), pp * cp, std::sqrt(s) / 2);
      TLorentzVector piMinus(-pp * tp * std::cos(az), -pp * tp * std::sin(az), -pp * cp, std::sqrt(s) / 2);
      piPlus.Boost(0, 0, -p / ePP);
      piMinus.Boost(0, 0, -p / ePP);
      std::array<TLorentzVector, 3> out = {{piPlus, piMinus, a}};
      orientAndBoost(out, parent, flat);
      return out;
    }

    // P -> pi+ pi- A (dark photon): s uniform, weight darkPhotonPiPiAWeight / bound (EtaDecayerModels.h).
    template <class Flat>
    std::array<TLorentzVector, 3> pipiADarkPhoton(
        const TLorentzVector& parent, double mA, double mLepA, const pipi::Context& c, double bound, Flat flat) {
      const double M = parent.M(), M2 = M * M, mA2 = mA * mA, mPi = c.mPi;
      const double smin = 4 * mPi * mPi, smax = std::pow(M - mA, 2), pi = std::acos(-1.);
      for (unsigned attempt = 0; attempt < 10 * kMaxAttempts; ++attempt) {
        const double s = smin + (smax - smin) * flat();
        const double cp = 2 * flat() - 1, cl = 2 * flat() - 1, phi = 2 * pi * flat();
        const double w = darkPhotonPiPiAWeight(s, cp, cl, phi, M2, mA2, mLepA, c) / bound;
        checkWeight(w, "P -> pi+ pi- A (dark photon)");
        if (flat() >= w)
          continue;
        const double p = splitMomentum(M, std::sqrt(s), mA);
        if (!(p > 0))
          continue;
        return pipiAKinematics(parent, M, s, cp, p, mA, mPi, flat);
      }
      throw cms::Exception("EtaDecayer") << "P -> pi+ pi- A (dark photon): rejection limit exhausted";
    }

    // P -> pi+ pi- A (ALP): three-body phase space, dGamma/ds ~ beta_pi p with s uniform, isotropic pi+ pi- decay.
    template <class Flat>
    std::array<TLorentzVector, 3> pipiAPhaseSpace(
        const TLorentzVector& parent, double mPi, double mA, double bound, Flat flat) {
      const double M = parent.M();
      const double smin = 4 * mPi * mPi, smax = std::pow(M - mA, 2);
      for (unsigned attempt = 0; attempt < kMaxAttempts; ++attempt) {
        const double s = smin + (smax - smin) * flat();
        const double bp = 1 - 4 * mPi * mPi / s;
        if (!(bp > 0))
          continue;
        const double p = splitMomentum(M, std::sqrt(s), mA);
        if (!(p > 0))
          continue;
        const double w = std::sqrt(bp) * p / bound;
        checkWeight(w, "P -> pi+ pi- A (ALP)");
        if (flat() >= w)
          continue;
        const double cp = 2 * flat() - 1;
        return pipiAKinematics(parent, M, s, cp, p, mA, mPi, flat);
      }
      throw cms::Exception("EtaDecayer") << "P -> pi+ pi- A (ALP): rejection limit exhausted";
    }

    // eta' -> eta pi1 pi2: s1 = m^2(eta pi1) and s2 = m^2(eta pi2) proposed uniformly (flat three-body phase space),
    // weight etapipi::weight / bound (EtaDecayerModels.h). Output: eta, pi1, pi2.
    template <class Flat>
    std::array<TLorentzVector, 3> etaPiPi(
        const TLorentzVector& parent, double mEta, double mPi, const etapipi::Params& params, double bound, Flat flat) {
      const double M = parent.M(), lo = std::pow(mEta + mPi, 2), hi = std::pow(M - mPi, 2), pi = std::acos(-1.);
      etapipi::Kinematics k;
      for (unsigned attempt = 0; attempt < kMaxAttempts; ++attempt) {
        const double s1 = lo + (hi - lo) * flat(), s2 = lo + (hi - lo) * flat();
        if (!etapipi::kinematics(M, mEta, mPi, s1, s2, k))
          continue;
        const double w = etapipi::weight(M, mEta, mPi, k, params) / bound;
        checkWeight(w, "eta' -> eta pi pi");
        if (flat() >= w)
          continue;
        // pi1 along z, pi2 in the xz plane, eta balancing; then a random rotation about z and orientAndBoost
        const double c = std::clamp((k.pEta * k.pEta - k.p1 * k.p1 - k.p2 * k.p2) / (2 * k.p1 * k.p2), -1., 1.);
        const double s = std::sqrt(1 - c * c);
        TLorentzVector pi1(0, 0, k.p1, k.e1), pi2(k.p2 * s, 0, k.p2 * c, k.e2);
        TLorentzVector eta(-(pi1 + pi2).Vect(), k.eEta);
        std::array<TLorentzVector, 3> out = {{eta, pi1, pi2}};
        const double az = 2 * pi * flat();
        for (auto& p : out)
          p.RotateZ(az);
        orientAndBoost(out, parent, flat);
        return out;
      }
      throw cms::Exception("EtaDecayer") << "eta' -> eta pi pi: rejection limit exhausted";
    }

  }  // namespace etadecayer
}  // namespace gen

#endif
