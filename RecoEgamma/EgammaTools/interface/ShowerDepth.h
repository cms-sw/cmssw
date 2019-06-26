#ifndef RecoEgamma_EgammaTools_ShowerDepth_h
#define RecoEgamma_EgammaTools_ShowerDepth_h

/* ShowerDepth: computes expected EM shower mean depth
 * in the HGCal, and compares it to measured depth
 *
 * Code copied from C. Charlot's
 * Hopefully correct explanations by N. Smith
 *
 * Based on gamma distribution description of electromagnetic cascades [PDG2016, ss33.5]
 * Basic equation:
 *   D_exp = X_0 t_max alpha / (alpha-1)
 * where D_exp is expectedDepth, X_0 the radiation length in HGCal material, t_max
 * the shower maximum and alpha a parameter of the gamma distribution. (beta := (a-1)/t_max)
 * Both t_max and alpha are approximated as first order polynomials of ln y = E/E_c, where E_c
 * is the critical energy, and presumably are extracted from fits to simulation, along with their
 * corresponding uncertainties.
 *
 * sigma(D_exp) then follows from error propagation and we can compare to
 * measured depth (distance between first hit and shower barycenter)
 *
 * The parameterization is for electrons, although photons will give similar results
 */

namespace hgcal {

  class ShowerDepth {
  public:
    ShowerDepth() {}
    ~ShowerDepth() {}

    float getClusterDepthCompatibility(float measuredDepth,
                                       float emEnergy,
                                       float& expectedDepth,
                                       float& expectedSigma) const;

  private:
    // HGCAL average medium
    static constexpr float criticalEnergy_ = 0.00536;  // in GeV
    static constexpr float radiationLength_ = 0.968;   // in cm

    // mean values
    // shower max <t_max> = t0 + t1*lny
    static constexpr float meant0_{-1.396};
    static constexpr float meant1_{1.007};
    // <alpha> = alpha0 + alpha1*lny
    static constexpr float meanalpha0_{-0.0433};
    static constexpr float meanalpha1_{0.540};
    // sigmas (relative uncertainty)
    // sigma(ln(t_max)) = 1 / (sigmalnt0 + sigmalnt1*lny);
    static constexpr float sigmalnt0_{-2.506};
    static constexpr float sigmalnt1_{1.245};
    // sigma(ln(alpha)) = 1 / (sigmalnt0 + sigmalnt1*lny);
    static constexpr float sigmalnalpha0_{-0.08442};
    static constexpr float sigmalnalpha1_{0.7904};
    // correlation coefficient
    // corr(ln(alpha), ln(t_max)) = corrlnalpha0_+corrlnalphalnt1_*lny
    static constexpr float corrlnalphalnt0_{0.7858};
    static constexpr float corrlnalphalnt1_{-0.0232};
  };

}  // namespace hgcal

#endif
