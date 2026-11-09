#ifndef GeneratorInterface_Pythia8Interface_EtaDecayerModels_h
#define GeneratorInterface_Pythia8Interface_EtaDecayerModels_h

// Physics inputs of EtaDecayer (EtaDecayer.h): transition form factors (TFF), the pi+ pi- gamma* amplitude, the
// dark-photon vertex factor, and the accept-reject bounds built from them. Only one model per vertex is
// implemented, chosen from a comparison of several published models with the published spectra:
//
//   P -> gamma(*) gamma* vertex, eta  : data-fitted Pade approximant P^7_1, Escribano, Masjuan, Sanchez-Puertas,
//                                       arXiv:1504.07742, Appendix A, Table 5
//   P -> gamma(*) gamma* vertex, eta' : Pade approximant P^6_1 below sqrt(q^2) = 0.70 GeV and rho+omega+phi vector
//                                       meson dominance (VMD) above, Escribano, Gonzalez-Solis, arXiv:1511.04916,
//                                       Sec. 2 (Pade coefficients from arXiv:1307.2061, Table IV)
//   double-virtual TFF                : factorized, F(q1^2, q2^2) = F(q1^2) F(q2^2) (arXiv:1511.04916, Sec. 2.4)
//   P -> pi+ pi- gamma* vertex        : Zillinger, Kubis, Sanchez-Puertas, arXiv:2210.14925, Eqs. (3.3)-(3.5), with
//                                       the P-wave Omnes function replaced by a rho Breit-Wigner with the
//                                       Eq. (3.5) energy-dependent width
//   eta' -> eta pi pi Dalitz plot     : BESIII general representation, arXiv:1709.04627, Table II
//
// All form factors are normalized to F(0) = 1; q2, s, sl are invariant masses squared in GeV^2.

#include "FWCore/Utilities/interface/Exception.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

namespace gen {
  namespace etadecayer {

    using Complex = std::complex<double>;

    // Kallen function lambda(1, x, y) for x, y = m^2/M^2, clipped at 0.
    inline double kallen(double x, double y) { return std::max(0., std::pow(1 - x - y, 2) - 4 * x * y); }

    // Momentum of A in the rest frame of a parent of mass M decaying to A (mass mA) + X (mass q).
    // Returns -1 outside the physical region. The factorized form is stable near threshold.
    inline double splitMomentum(double M, double q, double mA) {
      const double a = M * M - (q + mA) * (q + mA);
      const double b = M * M - (q - mA) * (q - mA);
      if (!(a > 0 && b > 0))
        return -1.;
      return std::sqrt(a * b) / (2 * M);
    }

    // ---------------------------------------------------------------------------------------------------------------
    // eta TFF: Pade approximant P^7_1 fitted to space-like data (arXiv:1504.07742, Table 5),
    //   Q^2 F(Q^2) = Q^2 (t1 + t2 Q^2 + ... + t7 Q^12) / (1 + r1 Q^2),  Q^2 = -q^2.
    // The pole sits at the time-like q^2 = 1/r1 = 0.511 GeV^2, above the eta range q^2 <= M_eta^2 = 0.300 GeV^2.
    // ---------------------------------------------------------------------------------------------------------------
    namespace eta_tff {
      constexpr double t1 = 0.27349, t2 = 1.1771e-2, t3 = -1.1048e-3, t4 = 2.8861e-5, t5 = 2.2974e-6, t6 = -1.5096e-7,
                       t7 = 2.3655e-9, r1 = 1.9584;

      inline double formFactor(double q2) {
        const double Q2 = -q2;
        const double num = Q2 * (t1 + Q2 * (t2 + Q2 * (t3 + Q2 * (t4 + Q2 * (t5 + Q2 * (t6 + Q2 * t7))))));
        const double den = 1 + r1 * Q2;
        if (!(den > 0))
          throw cms::Exception("EtaDecayer") << "eta TFF evaluated at or beyond its pole, q^2 = " << q2 << " GeV^2";
        if (std::abs(q2) < 1e-12)
          return t1;
        return num / den / Q2;  // num/den = Q^2 F(Q^2)
      }
    }  // namespace eta_tff

    // F(q^2)/F(0) for the eta.
    inline double etaFormFactorNorm(double q2) { return eta_tff::formFactor(q2) / eta_tff::formFactor(0.); }

    // Accept-reject bound for |F(q^2)|^2 over 4 m_l^2 <= q^2 <= M^2 (P -> gamma l+ l-, eta): the maximum on a
    // uniform grid times a safety margin of 1.2 (|F|^2 is monotonic in this range).
    inline double etaSingleBound(double mParent, double mLep, unsigned grid = 400) {
      const double low = 4 * mLep * mLep, high = mParent * mParent;
      double maxW = 0.;
      for (unsigned i = 0; i <= grid; ++i) {
        const double q2 = low + (high - low) * i / grid;
        maxW = std::max(maxW, std::pow(etaFormFactorNorm(q2), 2));
      }
      return 1.2 * maxW;
    }

    // Accept-reject bound for |F(s1)|^2 |F(s2)|^2 over sqrt(s1) + sqrt(s2) < M (eta double Dalitz decays):
    // the maximum on a uniform 2D grid times 1.2.
    inline double etaDoubleBound(double mParent, double mLep1, double mLep2, unsigned grid = 300) {
      const double low1 = 4 * mLep1 * mLep1, high1 = std::pow(mParent - 2 * mLep2, 2);
      const double low2 = 4 * mLep2 * mLep2, high2 = std::pow(mParent - 2 * mLep1, 2);
      double maxW = 0.;
      for (unsigned i = 0; i <= grid; ++i) {
        const double s1 = low1 + (high1 - low1) * i / grid;
        for (unsigned j = 0; j <= grid; ++j) {
          const double s2 = low2 + (high2 - low2) * j / grid;
          if (std::sqrt(s1) + std::sqrt(s2) >= mParent)
            continue;
          maxW = std::max(maxW, std::pow(etaFormFactorNorm(s1), 2) * std::pow(etaFormFactorNorm(s2), 2));
        }
      }
      return 1.2 * maxW;
    }

    // ---------------------------------------------------------------------------------------------------------------
    // eta' TFF over 0 <= q^2 <= M_eta'^2 (arXiv:1511.04916, Sec. 2). A single Pade approximant cannot be used: its
    // pole (q^2 = 1/r1 = 0.69 GeV^2) lies inside the eta' range. Two regimes, as in the paper:
    //   sqrt(q^2) <= 0.70 GeV: Pade P^6_1 of arXiv:1307.2061, Table IV (eta' column);
    //   sqrt(q^2) >  0.70 GeV: coherent rho + omega + phi VMD,
    //       F(q^2) = [sum_V w_V]^-1 sum_V w_V M_V^2 / (M_V^2 - q^2 - i M_V Gamma_V(q^2)),
    //     with the energy-dependent rho width of the paper's footnote and constant omega/phi widths, rescaled by one
    //     real constant so that |F| is continuous at 0.70 GeV (the paper's matching prescription).
    // VMD weights: arXiv:1511.04916 takes the couplings w_V = g_{V eta' gamma}/(2 g_{V gamma}) from the PDG without
    // quoting numbers. Here they are derived instead from the quark-flavour mixing scheme (mixing angle 39.6 deg,
    // arXiv:1307.2061 Sec. III) with ideal vector-meson flavour content and quark-charge photon couplings:
    // w_rho : w_omega : w_phi = sin(phi)/(2 sqrt2) : sin(phi)/(18 sqrt2) : cos(phi)/9. This choice sets the height
    // of the omega peak and the m_ll > 0.8 GeV tail: compared with the arXiv:1511.04916 eta' -> mu+ mu- gamma
    // spectrum the omega peak differs by 8% (at the resolution of the digitized curve), the tail above 0.8 GeV
    // (< 0.1% of the decays) is lower by up to a factor ~2, and below 0.74 GeV the agreement is within 1%.
    // Masses and widths are PDG 2016, the paper's source.
    // Only |F|^2 is defined: the real Pade piece and the complex VMD piece are matched in modulus only.
    // ---------------------------------------------------------------------------------------------------------------
    namespace etaprime_tff {
      constexpr double kPi = 3.14159265358979323846;
      constexpr double t1 = 0.343, t2 = 0.007, t3 = 0.986e-3, t4 = 0.744e-4, t5 = 0.252e-5, t6 = 0.290e-7, r1 = 1.442;
      constexpr double matchSqrtS = 0.70;  // GeV
      constexpr double matchQ2 = matchSqrtS * matchSqrtS;
      constexpr double mRho = 0.77526, gRho = 0.1491;
      constexpr double mOmega = 0.78265, gOmega = 0.00849;
      constexpr double mPhi = 1.019461, gPhi = 0.004266;
      constexpr double mPi = 0.13957;
      constexpr double mixingAngleDeg = 39.6;

      struct Weights {
        double rho, omega, phi;
      };
      inline const Weights& weights() {
        static const Weights w = [] {
          const double phi = mixingAngleDeg * kPi / 180., s = std::sin(phi), c = std::cos(phi), r2 = std::sqrt(2.);
          return Weights{s / (2 * r2), s / (18 * r2), c / 9};
        }();
        return w;
      }

      // Gamma_rho(q^2) = Gamma_rho (q^2/M_rho^2) sigma^3(q^2)/sigma^3(M_rho^2), zero below the two-pion threshold.
      inline double rhoWidth(double q2) {
        const double threshold = 4 * mPi * mPi;
        if (q2 <= threshold)
          return 0.;
        const double sigma = std::sqrt(1 - threshold / q2), sigma0 = std::sqrt(1 - threshold / (mRho * mRho));
        return gRho * (q2 / (mRho * mRho)) * std::pow(sigma / sigma0, 3);
      }

      inline double padeNorm(double q2) {
        const double Q2 = -q2;
        const double num = t1 + Q2 * (t2 + Q2 * (t3 + Q2 * (t4 + Q2 * (t5 + Q2 * t6))));
        const double den = 1 + r1 * Q2;
        if (!(den > 0))
          throw cms::Exception("EtaDecayer")
              << "eta' Pade TFF evaluated at or beyond its pole, q^2 = " << q2 << " GeV^2";
        return num / den / t1;
      }

      inline Complex vmdNorm(double q2) {
        const auto& w = weights();
        const auto term = [q2](double weight, double mass, double width) {
          return weight * mass * mass / Complex(mass * mass - q2, -mass * width);
        };
        return (term(w.rho, mRho, rhoWidth(q2)) + term(w.omega, mOmega, gOmega) + term(w.phi, mPhi, gPhi)) /
               (w.rho + w.omega + w.phi);
      }

      inline double matchScale() {
        static const double r = padeNorm(matchQ2) / std::abs(vmdNorm(matchQ2));
        return r;
      }
    }  // namespace etaprime_tff

    // |F(q^2)/F(0)|^2 for the eta'.
    inline double etaPrimeFormFactorNormSq(double q2) {
      using namespace etaprime_tff;
      if (q2 <= matchQ2) {
        const double f = padeNorm(q2);
        return f * f;
      }
      const double r = matchScale();
      return r * r * std::norm(vmdNorm(q2));
    }

    // Accept-reject bound for |F(q^2)|^2 over 4 m_l^2 <= q^2 <= M^2 (P -> gamma l+ l-, eta'): a dense grid that
    // resolves the narrow omega peak, a golden-section polish around the best grid point, and a margin of 1.05.
    inline double etaPrimeSingleBound(double mParent, double mLep, unsigned grid = 20000) {
      const double lo = 4 * mLep * mLep, hi = mParent * mParent;
      double best = 0., bestQ2 = lo;
      const double step = (hi - lo) / grid;
      for (unsigned i = 0; i <= grid; ++i) {
        const double q2 = lo + step * i, f = etaPrimeFormFactorNormSq(q2);
        if (f > best) {
          best = f;
          bestQ2 = q2;
        }
      }
      double a = std::max(lo, bestQ2 - step), b = std::min(hi, bestQ2 + step);
      const double gr = 0.6180339887498949;
      for (unsigned it = 0; it < 80; ++it) {
        const double c = b - gr * (b - a), d = a + gr * (b - a);
        if (etaPrimeFormFactorNormSq(c) > etaPrimeFormFactorNormSq(d))
          b = d;
        else
          a = c;
      }
      return 1.05 * std::max(best, etaPrimeFormFactorNormSq(0.5 * (a + b)));
    }

    // Accept-reject bound for |F(s1)|^2 |F(s2)|^2 over sqrt(s1) + sqrt(s2) < M (eta' double Dalitz decays): for each
    // s1 on the grid the best partner is the running maximum of |F(s2)|^2 over the allowed s2; margin 1.05.
    inline double etaPrimeDoubleBound(double mParent, double mLep1, double mLep2, unsigned grid = 20000) {
      const double low1 = 4 * mLep1 * mLep1, high1 = std::pow(mParent - 2 * mLep2, 2);
      const double low2 = 4 * mLep2 * mLep2, high2 = std::pow(mParent - 2 * mLep1, 2);
      std::vector<double> runMax2(grid + 1);
      double running = 0.;
      for (unsigned j = 0; j <= grid; ++j) {
        running = std::max(running, etaPrimeFormFactorNormSq(low2 + (high2 - low2) * j / grid));
        runMax2[j] = running;
      }
      double best = 0.;
      for (unsigned i = 0; i <= grid; ++i) {
        const double s1 = low1 + (high1 - low1) * i / grid;
        const double s2max = std::pow(mParent - std::sqrt(s1), 2);
        if (s2max <= low2)
          continue;
        const double pos = std::ceil((std::min(s2max, high2) - low2) / (high2 - low2) * grid);
        best = std::max(best,
                        etaPrimeFormFactorNormSq(s1) * runMax2[std::min<unsigned>(grid, static_cast<unsigned>(pos))]);
      }
      return 1.05 * best;
    }

    // |F(q^2)/F(0)|^2 of the parent (eta or eta').
    inline double formFactorNormSq(double q2, bool isEtaPrime) {
      return isEtaPrime ? etaPrimeFormFactorNormSq(q2) : std::pow(etaFormFactorNorm(q2), 2);
    }

    // ---------------------------------------------------------------------------------------------------------------
    // P -> pi+ pi- gamma* (arXiv:2210.14925, Sec. 3): f(s, sl) = P(s) Omega(s) Fbar(sl), s = m_pipi^2, sl = m_ll^2.
    //   P(s)    : polynomial fitted to eta(') -> pi+ pi- gamma data, Eq. (3.4) (for the eta' with the omega term)
    //   Omega(s): the P-wave Omnes function, here approximated by a rho Breit-Wigner normalized to 1 at s = 0
    //   Fbar(sl): coherent rho + rho(1450) dilepton form factor, Eq. (3.5)
    // Widths: Gamma_V(s) = Gamma_V sqrt(s)/M_V beta_pi^3(s)/beta_pi^3(M_V^2) theta(s - 4 m_pi^2), Eq. (3.5).
    // The rho and rho(1450) masses and widths are taken from the Pythia particle data (PDG ids 113 and 100113).
    // ---------------------------------------------------------------------------------------------------------------
    namespace pipi {
      struct Context {
        double mPi = 0.13957;
        bool isEtaPrime = false;
        double rhoMass = 0.77549, rhoGamma = 0.1491, rhopMass = 1.465, rhopGamma = 0.4;
      };

      inline Complex polynomial(double s, bool isEtaPrime) {
        if (!isEtaPrime) {
          constexpr double A = 17.9, alpha = 1.52;  // GeV^-3, GeV^-2
          return A * (1 + alpha * s);
        }
        constexpr double A = 16.7, alpha = 1.00, beta = -0.55, kappa2 = 6.72e-3;
        constexpr double mOmega = 0.78266, gammaOmega = 0.00868;
        const Complex omegaPole = kappa2 / Complex(mOmega * mOmega - s, -mOmega * gammaOmega);
        return A * (1.0 + alpha * s + beta * s * s + omegaPole);
      }

      inline double width(double s, double mV, double gV, double mPi) {
        if (s <= 4 * mPi * mPi)
          return 0.;
        const double b = std::sqrt(1 - 4 * mPi * mPi / s), bV = std::sqrt(1 - 4 * mPi * mPi / (mV * mV));
        return gV * std::sqrt(s) / mV * std::pow(b / bV, 3);
      }

      inline Complex fbar(double sl, const Context& c) {
        const Complex dR(c.rhoMass * c.rhoMass - sl, -std::sqrt(sl) * width(sl, c.rhoMass, c.rhoGamma, c.mPi));
        const Complex dP(c.rhopMass * c.rhopMass - sl, -std::sqrt(sl) * width(sl, c.rhopMass, c.rhopGamma, c.mPi));
        return c.rhoMass * c.rhoMass * c.rhopMass * c.rhopMass / (dR * dP);
      }

      inline Complex omegaBW(double s, const Context& c) {
        return c.rhoMass * c.rhoMass /
               Complex(c.rhoMass * c.rhoMass - s, -std::sqrt(s) * width(s, c.rhoMass, c.rhoGamma, c.mPi));
      }

      // |f(s, sl)|^2 up to a constant.
      inline double massWeight(double s, double sl, const Context& c) {
        return std::norm(polynomial(s, c.isEtaPrime)) * std::norm(omegaBW(s, c)) * std::norm(fbar(sl, c));
      }
    }  // namespace pipi

    // Accept-reject bound for pipi::massWeight over the physical (s, sl) region of P -> pi+ pi- l+ l-:
    // the maximum on a uniform 2D grid times 1.2.
    inline double llpipiBound(double mParent, double mLep, const pipi::Context& c, unsigned grid = 300) {
      const double mpi = c.mPi;
      const double lowS = 4 * mpi * mpi, highS = std::pow(mParent - 2 * mLep, 2);
      const double lowSl = 4 * mLep * mLep, highSl = std::pow(mParent - 2 * mpi, 2);
      double maxW = 0.;
      for (unsigned i = 0; i <= grid; ++i) {
        const double s = lowS + (highS - lowS) * i / grid;
        for (unsigned j = 0; j <= grid; ++j) {
          const double sl = lowSl + (highSl - lowSl) * j / grid;
          if (std::sqrt(s) + std::sqrt(sl) >= mParent)
            continue;
          maxW = std::max(maxW, pipi::massWeight(s, sl, c));
        }
      }
      return 1.2 * maxW;
    }

    // ---------------------------------------------------------------------------------------------------------------
    // eta' -> eta pi pi Dalitz plot (BESIII, arXiv:1709.04627, Eqs. (1)-(3) and Table II), general representation
    //   |M(X, Y)|^2 = N (1 + a Y + b Y^2 + d X^2),
    //   X = sqrt(3) (T_pi1 - T_pi2) / Q,  Y = (m_eta + 2 m_pi) / m_pi * T_eta / Q - 1,  Q = M - m_eta - 2 m_pi,
    // with T the kinetic energies in the eta' rest frame, pi1 = pi+ for the charged mode, and m_pi the mass of the
    // pions of the mode (charged or neutral). Odd powers of X vanish (C invariance for pi+ pi-, Bose symmetry for
    // pi0 pi0); the c X term is set to zero, as in the nominal BESIII fits. Not included: higher-order terms (fitted
    // consistent with zero) and the pi+ pi- threshold cusp in eta pi0 pi0 (a few-percent effect near the threshold,
    // arXiv:2207.01004).
    // ---------------------------------------------------------------------------------------------------------------
    namespace etapipi {
      struct Params {
        double a, b, d;
      };
      constexpr Params kCharged{-0.056, -0.049, -0.063};  // eta' -> eta pi+ pi-
      constexpr Params kNeutral{-0.087, -0.073, -0.074};  // eta' -> eta pi0 pi0

      // Rest-frame kinematics of M -> eta pi1 pi2 from s1 = m^2(eta pi1), s2 = m^2(eta pi2).
      struct Kinematics {
        double e1, e2, eEta;  // energies of pi1, pi2, eta
        double p1, p2, pEta;  // momentum magnitudes
      };

      // False outside the Dalitz region.
      inline bool kinematics(double M, double mEta, double mPi, double s1, double s2, Kinematics& k) {
        k.e2 = (M * M + mPi * mPi - s1) / (2 * M);
        k.e1 = (M * M + mPi * mPi - s2) / (2 * M);
        k.eEta = M - k.e1 - k.e2;
        if (!(k.e1 > mPi && k.e2 > mPi && k.eEta > mEta))
          return false;
        k.p1 = std::sqrt(k.e1 * k.e1 - mPi * mPi);
        k.p2 = std::sqrt(k.e2 * k.e2 - mPi * mPi);
        k.pEta = std::sqrt(k.eEta * k.eEta - mEta * mEta);
        return k.pEta <= k.p1 + k.p2 && k.pEta >= std::abs(k.p1 - k.p2);
      }

      // |M(X, Y)|^2 / N
      inline double weight(double M, double mEta, double mPi, const Kinematics& k, const Params& p) {
        const double Q = M - mEta - 2 * mPi;
        const double X = std::sqrt(3.) * (k.e1 - k.e2) / Q;
        const double Y = (mEta + 2 * mPi) / mPi * (k.eEta - mEta) / Q - 1;
        return 1 + p.a * Y + p.b * Y * Y + p.d * X * X;
      }
    }  // namespace etapipi

    // Accept-reject bound for etapipi::weight over the Dalitz plot: the maximum on a uniform (s1, s2) grid times 1.05
    // (the weight is a smooth quadratic in X, Y and varies by less than 30% over the plot).
    inline double etaPiPiBound(double mParent, double mEta, double mPi, const etapipi::Params& p, unsigned grid = 400) {
      const double lo = std::pow(mEta + mPi, 2), hi = std::pow(mParent - mPi, 2);
      double maxW = 0.;
      etapipi::Kinematics k;
      for (unsigned i = 0; i <= grid; ++i)
        for (unsigned j = 0; j <= grid; ++j) {
          const double s1 = lo + (hi - lo) * i / grid, s2 = lo + (hi - lo) * j / grid;
          if (etapipi::kinematics(mParent, mEta, mPi, s1, s2, k))
            maxW = std::max(maxW, etapipi::weight(mParent, mEta, mPi, k, p));
        }
      return 1.05 * maxW;
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Dark photon A (kinetic mixing: A couples like a massive photon).
    // ---------------------------------------------------------------------------------------------------------------
    // P -> gamma* A vertex factor (2p/M)^3 = lambda^{3/2}(1, x, r), x = q^2/M^2, r = m_A^2/M^2: the spin-summed square
    // of the epsilon-tensor vertex times the P -> A + gamma* two-body phase space. Exact for any m_A; it reduces to the
    // real-photon factor (1-x)^3 for m_A -> 0. Factorized form for numerical stability near the endpoint.
    inline double darkPhotonVertexFactor(double x, double r) {
      const double sx = std::sqrt(x), sr = std::sqrt(r);
      const double l = (1 - (sx + sr) * (sx + sr)) * (1 - (sx - sr) * (sx - sr));
      return l > 0 ? l * std::sqrt(l) : 0.;
    }

    // Accept-reject bound for |F(q^2)|^2 in P -> l+ l- A (dark photon): maximum on a 2000-point logarithmic x grid
    // times 1.2. The vertex and angular factors are each <= 1.
    inline double darkPhotonLLABound(double mParent, double mLep, double mA, bool isEtaPrime, unsigned grid = 2000) {
      const double M2 = mParent * mParent;
      const double xlo = 4 * mLep * mLep / M2, xhi = (mParent - mA) * (mParent - mA) / M2;
      double maxW = 0.;
      for (unsigned i = 0; i <= grid; ++i) {
        const double x = xlo * std::exp(std::log(xhi / xlo) * i / grid);
        maxW = std::max(maxW, formFactorNormSq(x * M2, isEtaPrime));
      }
      return 1.2 * maxW;
    }

    // Weight of P -> pi+ pi- A (dark photon): the P -> pi+ pi- l+ l- density (phase space, magnetic angular term and
    // pipi::massWeight) with the dilepton mass fixed to m_A. s = m_pipi^2, cp = cos(theta_pi); cl and phi are the
    // dilepton angles of the four-body formula, sampled and then discarded: their factor does not depend on s, so it
    // only changes the efficiency. mLepA is the lepton mass entering that factor (the muon, A -> mu+ mu-), which
    // requires m_A > 2 mLepA.
    inline double darkPhotonPiPiAWeight(
        double s, double cp, double cl, double phi, double M2, double mA2, double mLepA, const pipi::Context& c) {
      const double bp = 1 - 4 * c.mPi * c.mPi / s;
      if (!(bp > 0))
        return 0.;
      const double lam = kallen(s / M2, mA2 / M2);
      const double bl = 1 - 4 * mLepA * mLepA / mA2;
      const double ang = 1 - bl * (1 - cl * cl) * std::pow(std::sin(phi), 2);
      return std::pow(lam, 1.5) * (s / M2) * std::pow(bp, 1.5) * std::sqrt(bl) * (1 - cp * cp) * ang *
             pipi::massWeight(s, mA2, c);
    }

    // Accept-reject bound for darkPhotonPiPiAWeight: maximum over a uniform s grid at the angular maximum, times 1.2.
    inline double darkPhotonPiPiABound(
        double mParent, double mA, double mLepA, const pipi::Context& c, unsigned grid = 4000) {
      const double M2 = mParent * mParent, mA2 = mA * mA, smin = 4 * c.mPi * c.mPi, smax = std::pow(mParent - mA, 2);
      double maxW = 0.;
      for (unsigned i = 0; i <= grid; ++i) {
        const double s = smin + (smax - smin) * i / grid;
        maxW = std::max(maxW, darkPhotonPiPiAWeight(s, 0., 0., 0., M2, mA2, mLepA, c));
      }
      return 1.2 * maxW;
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Axion-like particle A (pseudoscalar): no production amplitude is implemented; P -> l+ l- A and P -> pi+ pi- A
    // follow three-body phase space. The bounds are the phase-space densities' maxima on a uniform grid times 1.2.
    // ---------------------------------------------------------------------------------------------------------------
    // P -> l+ l- A: dGamma/dx ~ beta(x) p(x) for x = q^2/M^2; the sampler proposes x log-uniformly (density 1/x), so
    // the weight is beta p x.
    inline double alpLLABound(double mParent, double mLep, double mA, unsigned grid = 4000) {
      const double M2 = mParent * mParent;
      const double xlo = 4 * mLep * mLep / M2, xhi = (mParent - mA) * (mParent - mA) / M2;
      double maxW = 0.;
      for (unsigned i = 0; i <= grid; ++i) {
        const double x = xlo + (xhi - xlo) * i / grid;
        const double beta2 = 1 - 4 * mLep * mLep / (x * M2);
        if (!(beta2 > 0))
          continue;
        const double p = splitMomentum(mParent, mParent * std::sqrt(x), mA);
        if (!(p > 0))
          continue;
        maxW = std::max(maxW, std::sqrt(beta2) * p * x);
      }
      return 1.2 * maxW;
    }

    // P -> pi+ pi- A: dGamma/ds ~ beta_pi(s) p(s), with s proposed uniformly.
    inline double alpPiPiABound(double mParent, double mPi, double mA, unsigned grid = 4000) {
      const double smin = 4 * mPi * mPi, smax = std::pow(mParent - mA, 2);
      double maxW = 0.;
      for (unsigned i = 0; i <= grid; ++i) {
        const double s = smin + (smax - smin) * i / grid;
        const double bp = 1 - 4 * mPi * mPi / s;
        if (!(bp > 0))
          continue;
        const double p = splitMomentum(mParent, std::sqrt(s), mA);
        if (!(p > 0))
          continue;
        maxW = std::max(maxW, std::sqrt(bp) * p);
      }
      return 1.2 * maxW;
    }

  }  // namespace etadecayer
}  // namespace gen

#endif
