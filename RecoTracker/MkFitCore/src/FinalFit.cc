#include "FinalFit.h"
#include "PlaneSteps.h"

#include "PropagationMPlex.h"

#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/PropagationConfig.h"
#include "RecoTracker/MkFitCore/interface/cms_common_macros.h"

#include <vdt/atan2.h>
#include <cstring>

namespace mkfit::final_fit {

  using namespace plane;

  namespace {

    using MPF = MPlexQF;

    // ---- vdt::fast_atan2f over a whole Matriplex, vectorised whatever the floating-point flags.
    // With trapping math GCC does not if-convert the branches inside vdt::fast_atan2f, and Matriplex::fast_atan2 runs
    // lane by lane ("not vectorized: control flow in loop").  Here the same operations, in the same order, are written
    // with GCC vector extensions, every branch an explicit blend of values that are all computed.  Bit-identical to
    // Matriplex::fast_atan2 on 1e9 inputs (log-uniform 1e-6..1e3, both signs, zeros, y ~ x).
    typedef float vfloat __attribute__((vector_size(NN * sizeof(float))));
    typedef int vint __attribute__((vector_size(NN * sizeof(int))));

    MPF atan2V(const MPF& Y, const MPF& X) {
      vfloat y, x;
      std::memcpy(&y, Y.fArray, sizeof(y));
      std::memcpy(&x, X.fArray, sizeof(x));
      const vfloat zero = vfloat{} + 0.f, one = vfloat{} + 1.f;
      const vfloat ax = (vfloat)((vint)x & 0x7fffffff), ay = (vfloat)((vint)y & 0x7fffffff);
      const vint swp = ay > ax;
      const vfloat xx = swp ? ay : ax;
      const vfloat yy = swp ? ax : ay;
      const vfloat oneIfXXZero = (xx == zero) ? one : zero;
      const vfloat t = yy / xx;
      const vint red = t > 0.4142135623730950f;
      const vfloat z = red ? (t - 1.0f) / (t + 1.0f) : t;
      const vfloat z2 = z * z;
      vfloat ret =
          ((((8.05374449538e-2f * z2 - 1.38776856032E-1f) * z2 + 1.99777106478E-1f) * z2 - 3.33329491539E-1f) * z2 * z +
           z);
      ret *= (1.f - oneIfXXZero);
      ret = (y == zero) ? zero : ret;
      ret = red ? ret + vdt::details::PIO4F : ret;
      ret = swp ? vdt::details::PIO2F - ret : ret;
      ret = (x < zero) ? vdt::details::PIF - ret : ret;
      ret = (y < zero) ? -ret : ret;
      MPF R;
      std::memcpy(R.fArray, &ret, sizeof(ret));
      return R;
    }

    // ---- Radial-field correction (FinalFitFlags::radial_field_corr) ----------------------------------
    //
    // The propagation models B as purely along z and constant over the step.  The real solenoid field is
    // axially symmetric with Br = -(r/2) dBz/dz (div B = 0), and such a field conserves the canonical
    // angular momentum
    //     L_z = r * p_phi + q * k * Psi(r,z) / (2 pi)
    // with Psi the flux enclosed at (r,z).  The constant-Bz helix conserves the same quantity with the
    // uniform flux q*k*Bc*r^2/2, so over a step it misses a change D of r*p_phi.  For the parametrised
    // field Bz = Z(z) (a r^2 + 1), Z = b0 z^2 + b1 z + c1, the flux is analytic,
    //     G = q*k*Z(z)*(a r^4/4 + r^2/2),
    // so no extra field lookup and no new constant is needed.  p_r is untouched and |p| is conserved, so
    // pT and theta both move -- the degree of freedom the constant-Bz helix freezes.
    //
    // D is exactly antisymmetric under swapping the step's endpoints.  It is applied as two half-kicks,
    // D/(2 r0) before the helix step and D/(2 r1) after it, each converted with the radius where it is
    // applied: reversing the step sends D -> -D and swaps r0 <-> r1, so the step stays its own inverse
    // (a one-sided D/r1 is not, and makes the two refit passes disagree).

    // D = d(r*p_phi) over one step for every lane, B field midpoint Bc = Zmid (a rmid^2 + 1).
    //   D/qk = D2*[ Zm*a*S/4 + (Zm - Bc)/2 ] + dZ*(f0 + f1)/2,  D2 = r0^2 - r1^2,  S = r0^2 + r1^2,
    //   f = a r^4/4 + r^2/2,  Zm = (Z0 + Z1)/2,  dZ = Z0 - Z1.
    // D2 and dZ flip sign under the swap and nothing else does.  Every difference is written in factored
    // form so that no two large, nearly equal numbers are subtracted in float.
    MPlexQF brDeltaRPphiV(const MPlexLV& inPar, const MPlexLV& outPar, const MPlexQI& inChg) {
      namespace mpt = Matriplex;
      using MPF = MPlexQF;
      const MPF x0 = inPar(0, 0), y0 = inPar(1, 0), z0 = inPar(2, 0);
      const MPF x1 = outPar(0, 0), y1 = outPar(1, 0), z1 = outPar(2, 0);
      const MPF r0sq = x0 * x0 + y0 * y0;
      const MPF r1sq = x1 * x1 + y1 * y1;
      // q * sol_over_100, without materialising q: +sol where the charge is positive
      const MPF qk = mpt::negate_if_ltz(MPF(Const::sol_over_100), inChg);
      const MPF Z0 = (MPF(Config::mag_fit_b0) * z0 + MPF(Config::mag_fit_b1)) * z0 + MPF(Config::mag_fit_c1);
      const MPF Z1 = (MPF(Config::mag_fit_b0) * z1 + MPF(Config::mag_fit_b1)) * z1 + MPF(Config::mag_fit_c1);
      const MPF dz = z0 - z1;
      const MPF dZ = dz * (MPF(Config::mag_fit_b0) * (z0 + z1) + MPF(Config::mag_fit_b1));
      const MPF zmid = 0.5f * (z0 + z1);
      const MPF xm = 0.5f * (x0 + x1), ym = 0.5f * (y0 + y1);
      const MPF rmid2 = xm * xm + ym * ym;
      const MPF Zmid = (MPF(Config::mag_fit_b0) * zmid + MPF(Config::mag_fit_b1)) * zmid + MPF(Config::mag_fit_c1);
      const MPF Zm = 0.5f * (Z0 + Z1);
      // (Zm - Bc) in factored form: Zm - Zmid = b0 dz^2 / 4
      const MPF ZmB = 0.25f * MPF(Config::mag_fit_b0) * dz * dz - Zmid * MPF(Config::mag_fit_a) * rmid2;
      const MPF D = r0sq - r1sq;
      const MPF S = r0sq + r1sq;
      const MPF f0 = 0.25f * MPF(Config::mag_fit_a) * r0sq * r0sq + 0.5f * r0sq;
      const MPF f1 = 0.25f * MPF(Config::mag_fit_a) * r1sq * r1sq + 0.5f * r1sq;
      return qk * (D * (0.25f * Zm * MPF(Config::mag_fit_a) * S + 0.5f * ZmB) + 0.5f * dZ * (f0 + f1));
    }

    // Rotate p_phi by a kick half_d / r at every lane's own position r, keeping p_r and |p|.  Vectorised
    // (vdt sincos via Matriplex, atan2V); lanes whose guards fail are left untouched.
    void applyDpPhiV(MPlexLV& par, const MPlexQF& half_d, const int N_proc) {
      namespace mpt = Matriplex;
      using MPF = MPlexQF;
      const MPF x = par(0, 0), y = par(1, 0);
      const MPF rsq = x * x + y * y;
      const MPF ipt = par(3, 0);

      // Clamp the guarded quantities so every lane can be evaluated; failing lanes are
      // discarded at write-back, exactly as the scalar version leaves them untouched.
      MPF rsq_s = rsq, ipt_s = ipt;
      for (int n = 0; n < NN; ++n) {
        if (!(rsq_s[n] > 1.e-8f))
          rsq_s[n] = 1.f;
        if (!(std::abs(ipt_s[n]) > 1.e-9f))
          ipt_s[n] = 1.f;
      }

      MPF sinP, cosP, sinT, cosT;
      mpt::fast_sincos(par(4, 0), sinP, cosP);
      mpt::fast_sincos(par(5, 0), sinT, cosT);
      MPF sinT_s = sinT;
      for (int n = 0; n < NN; ++n)
        if (!(std::abs(sinT_s[n]) > 1.e-9f))
          sinT_s[n] = 1.f;

      const MPF pt = mpt::negate_if_ltz(MPF(1.f) / ipt_s, ipt_s);  // = 1/|ipt|
      const MPF px = pt * cosP, py = pt * sinP;
      const MPF ptot = pt / sinT_s;
      const MPF pz = ptot * cosT;
      const MPF r = mpt::sqrt(rsq_s);
      const MPF invr = MPF(1.f) / r;
      const MPF dpphi = half_d / r;
      const MPF pr = (x * px + y * py) * invr;
      const MPF pphi = (x * py - y * px) * invr + dpphi;
      const MPF pt_new = mpt::sqrt(pr * pr + pphi * pphi);
      const MPF a2 = ptot * ptot - pt_new * pt_new;

      MPF a2_s = a2;
      for (int n = 0; n < NN; ++n)
        if (!(a2_s[n] > 0.f))
          a2_s[n] = 1.f;
      const MPF newphi = atan2V((y * pr + x * pphi) * invr, (x * pr - y * pphi) * invr);
      MPF pzn = mpt::sqrt(a2_s);
      for (int n = 0; n < NN; ++n)
        pzn[n] = std::copysign(pzn[n], pz[n]);
      const MPF newtheta = atan2V(pt_new, pzn);
      const MPF newipt = MPF(1.f) / pt_new;

      for (int n = 0; n < N_proc; ++n) {
        if (!(rsq[n] > 1.e-8f) || !(std::abs(ipt[n]) > 1.e-9f) || !(std::abs(sinT[n]) > 1.e-9f))
          continue;
        if (!(a2[n] > 0.f) || !(pt_new[n] > 1.e-9f))
          continue;
        par.At(n, 3, 0) = newipt[n];
        par.At(n, 4, 0) = newphi[n];
        par.At(n, 5, 0) = newtheta[n];
      }
    }

    // ---- Steps of the final fit's propagation (see PlaneSteps.h for the shared ones)

    // The field at the chord midpoint between the start state and at: B and kinv for the rest of the step.
    void field_at_mid(const TrackRef& in, const MPlexLV& at, FieldAt& f) {
      const MPF z = 0.5f * (in.par(2, 0) + at(2, 0));
      const MPF x = 0.5f * (in.par(0, 0) + at(0, 0));
      const MPF y = 0.5f * (in.par(1, 0) + at(1, 0));
      for (int n = 0; n < NN; ++n)
        f.b[n] = Config::bFieldFromZRFit(z[n], hipo(x[n], y[n]));
      f.kinv = Matriplex::negate_if_ltz(MPF(-Const::sol_over_100), in.chg) * f.b;
    }

    // A helix step from in to the plane pl: the path solve, with the field re-sampled at the chord midpoint
    // after each drift of the refinement when ffflags.b_field_at_mid, then the drift onto the plane and, if J
    // is given, its Jacobian, in the field the step was taken in.
    void helix_step(const TrackRef& in,
                    const PlaneRef& pl,
                    const PropagationFlags& pflags,
                    const FinalFitFlags& ffflags,
                    MPlexLV& outPar,
                    MPlexQF& s,
                    MPlexLL* J) {
      FieldAt f;
      field_at_start(in, pflags, f);
      StartTrig t;
      start_trig(in, t);
      PathSolve p;
      path_init(in, t, pl, f, p);
      // Where the first solve has no root, the refinement starts from the straight line (0 where that is not
      // finite either): a drift by a non-finite path length would make the midpoint field, and with it the whole
      // step, non-finite, beyond what the straight-line fallback of path_close() restores.
      for (int n = 0; n < in.n_proc; ++n)
        if (!mkfit::isFinite(p.s[n]))
          p.s[n] = mkfit::isFinite(p.s_line[n]) ? p.s_line[n] : 0.f;
      for (int i = 0; i < Config::nSStepsInProp2Plane - 1; ++i) {
        MPlexLV at{0.0f};
        drift(in, t, f, p.s, at);
        if (pflags.use_param_b_field && ffflags.b_field_at_mid)
          field_at_mid(in, at, f);
        path_refine_from(in, t, pl, f, at, p);
      }
      path_close(in, p);
      drift(in, t, f, p.s, outPar);
      if (J)
        jacobian(in, t, outPar, f, p.s, *J);
      s = p.s;
    }

    // The helix step with the radial-field correction when ffflags.radial_field_corr: D from the uncorrected
    // step, half of it as a kick at the start, the helix step from the kicked state, the other half at the end.
    // The Jacobian is that of the helix step: it describes the transport, not the correction, which is a
    // first-order effect on the weighting and not on the mean.
    void step_to_plane(const TrackRef& in,
                       const PlaneRef& pl,
                       const PropagationFlags& pflags,
                       const FinalFitFlags& ffflags,
                       MPlexLV& outPar,
                       MPlexQF& s,
                       MPlexLL* J) {
      if (pflags.use_param_b_field && ffflags.radial_field_corr) {
        MPlexLV par0{0.0f};
        MPlexQF s0{0.0f};
        helix_step(in, pl, pflags, ffflags, par0, s0, nullptr);
        const MPF halfD = 0.5f * brDeltaRPphiV(in.par, par0, in.chg);
        MPlexLV parH = in.par;
        applyDpPhiV(parH, halfD, in.n_proc);
        helix_step(TrackRef{parH, in.chg, in.n_proc}, pl, pflags, ffflags, outPar, s, J);
        applyDpPhiV(outPar, halfD, in.n_proc);
      } else {
        helix_step(in, pl, pflags, ffflags, outPar, s, J);
      }
    }

    // The end of a propagation to a plane: transport the covariance with J, apply the material at the destination
    // (the energy-loss sign from the pass, or from the path length s), bring phi into range, and restore the start
    // state on lanes that failed.
    void finish_on_plane(const TrackRef& in,
                         const MPlexLS& inErr,
                         const MPlexHV& plNrm,
                         const MPlexLL& J,
                         const MPlexQF& s,
                         MPlexLS& outErr,
                         MPlexLV& outPar,
                         MPlexQI& outFailFlag,
                         const Pass& pass,
                         const MPlexQI* noMatEffPtr,
                         const ModuleMaterial* modMat) {
      transport_cov(J, inErr, outErr);

      if (pass.pflags.apply_material) {
        MaterialAt m;
        if (pass.ffflags.material_per_module && modMat)
          material_given(modMat->radl, modMat->bbxi, noMatEffPtr, in.n_proc, m);
        else
          material_grid(*pass.pflags.tracker_info, outPar, noMatEffPtr, in.n_proc, m);
        if (pass.ffflags.eloss_sign_from_pass)
          eloss_sign_of_pass(pass.outward, noMatEffPtr, in.n_proc, m);
        else
          eloss_sign_from_path(s, noMatEffPtr, in.n_proc, m);
        apply_material(m, plNrm, outErr, outPar, in.n_proc, pass.ms_ref_p);
      }

      finish(in, inErr, outFailFlag, outPar, outErr);
    }

    // The first path-length estimate to the plane, the start of the path solve; the straight line where that is
    // not finite, and zero where neither is.  It only partitions a step into sub-steps.
    MPF first_path_estimate(const TrackRef& in, const PlaneRef& pl, const PropagationFlags& pflags) {
      FieldAt f;
      field_at_start(in, pflags, f);
      StartTrig t;
      start_trig(in, t);
      PathSolve p;
      path_init(in, t, pl, f, p);
      MPF s0{0.0f};
      for (int n = 0; n < in.n_proc; ++n) {
        float s = p.s[n];
        if (!mkfit::isFinite(s))
          s = p.s_line[n];
        s0[n] = mkfit::isFinite(s) ? s : 0.f;
      }
      return s0;
    }

    // One sub-step of fixed path length step (per lane; 0 = no move), parameters only, with the field model of a
    // full step.
    void fixed_length_sub_step(MPlexLV& par,
                               const MPlexQI& chg,
                               const MPF& step,
                               const int N_proc,
                               const PropagationFlags& pflags,
                               const FinalFitFlags& ffflags) {
      const TrackRef in{par, chg, N_proc};
      // a drift with B at the start
      FieldAt f;
      field_at_start(in, pflags, f);
      StartTrig t;
      start_trig(in, t);
      MPlexLV p1{0.0f};
      drift(in, t, f, step, p1);
      // b_field_at_mid: the drift again, with B at the chord midpoint
      if (pflags.use_param_b_field && ffflags.b_field_at_mid) {
        field_at_mid(in, p1, f);
        drift(in, t, f, step, p1);
      }
      // radial_field_corr: half of the radial-field kick D before the drift and half after it, D between the
      // uncorrected endpoints; the kicked drift keeps the midpoint field, which the kick moves only at second order
      if (pflags.use_param_b_field && ffflags.radial_field_corr) {
        const MPF dvec = brDeltaRPphiV(par, p1, chg);
        MPlexLV ph = par;
        applyDpPhiV(ph, 0.5f * dvec, N_proc);
        const TrackRef inh{ph, chg, N_proc};
        StartTrig th;
        start_trig(inh, th);
        drift(inh, th, f, step, p1);
        applyDpPhiV(p1, 0.5f * dvec, N_proc);
      }
      squashPhiMPlex(p1, N_proc);
      par = p1;
    }

    // The Kalman update in the local frame, with the covariance in Joseph form, as KFUpdator:
    //   C' = (I - K H) C (I - K H)^T + K V K^T,
    // V the hit's local covariance.  Positive semi-definite by construction, also in single precision, where
    // (I - K H) C loses definiteness when a hit shrinks a variance a lot.  H picks the local position (parameters 3
    // and 4), which the products below use directly.  The gain and the parameters are computed as in plane::update().
    // Row pointers are taken outside the lane loops: through the offset table of a symmetric Matriplex inside them,
    // the addresses are not affine and the loops do not vectorise.
    void update_joseph(const LocalPred& L, const LocalMeas& M, const Residual& R, LocalUpd& U) {
      constexpr int D = 5;
      const MPlex5S& predErr = L.err;
      // K, K V and (I - K H) C, in the Matriplex layout [row][column][lane] as plain arrays: GCC vectorises the lane
      // loops over these, not over Matriplex accessors in the same loop nests
      float gain[D][2][NN], gainV[D][2][NN], fwdTerm[D][D][NN];
      const float* sInv00 = &R.s_inv.constAt(0, 0, 0);
      const float* sInv01 = &R.s_inv.constAt(0, 0, 1);
      const float* sInv11 = &R.s_inv.constAt(0, 1, 1);
      const float* hitV00 = &M.err.constAt(0, 0, 0);
      const float* hitV01 = &M.err.constAt(0, 0, 1);
      const float* hitV11 = &M.err.constAt(0, 1, 1);
      const float* res0 = &R.r.constAt(0, 0, 0);
      const float* res1 = &R.r.constAt(0, 1, 0);
      for (int i = 0; i < D; ++i) {
        const float* cI3 = &predErr.constAt(0, i, 3);
        const float* cI4 = &predErr.constAt(0, i, 4);
        const float* parI = &L.lp.constAt(0, i, 0);
        float* updParI = &U.lp.At(0, i, 0);
#pragma omp simd
        for (int n = 0; n < NN; ++n) {
          gain[i][0][n] = sInv00[n] * cI3[n] + sInv01[n] * cI4[n];
          gain[i][1][n] = sInv01[n] * cI3[n] + sInv11[n] * cI4[n];
          updParI[n] = parI[n] + gain[i][0][n] * res0[n] + gain[i][1][n] * res1[n];
          gainV[i][0][n] = gain[i][0][n] * hitV00[n] + gain[i][1][n] * hitV01[n];
          gainV[i][1][n] = gain[i][0][n] * hitV01[n] + gain[i][1][n] * hitV11[n];
        }
      }
      for (int j = 0; j < D; ++j) {
        const float* c3J = &predErr.constAt(0, 3, j);
        const float* c4J = &predErr.constAt(0, 4, j);
        for (int i = 0; i < D; ++i) {
          const float* cIJ = &predErr.constAt(0, i, j);
#pragma omp simd
          for (int n = 0; n < NN; ++n)
            fwdTerm[i][j][n] = cIJ[n] - gain[i][0][n] * c3J[n] - gain[i][1][n] * c4J[n];
        }
      }
      // C' = (I - K H) C (I - K H)^T + K V K^T
      for (int i = 0; i < D; ++i)
        for (int j = 0; j <= i; ++j) {
          float* updErrIJ = &U.err.At(0, i, j);
#pragma omp simd
          for (int n = 0; n < NN; ++n)
            updErrIJ[n] = fwdTerm[i][j][n] - fwdTerm[i][3][n] * gain[j][0][n] - fwdTerm[i][4][n] * gain[j][1][n] +
                          gainV[i][0][n] * gain[j][0][n] + gainV[i][1][n] * gain[j][1][n];
        }
    }

    // The Kalman update of the predicted state (psErr, psPar) with the hits, and its chi2.
    void update_on_plane(const MPlexLS& psErr,
                         const MPlexLV& psPar,
                         const MPlexQI& inChg,
                         const MPlexHS& msErr,
                         const MPlexHV& msPar,
                         const MPlexHV& plNrm,
                         const MPlexHV& plDir,
                         const MPlexHV& plPnt,
                         MPlexLS& outErr,
                         MPlexLV& outPar,
                         MPlexQF& outChi2,
                         const int N_proc,
                         const PropagationFlags& pflags,
                         const MPlexQI* doCPE,
                         cpe_func cpe_corr_func,
                         const LocalStatesOut* localStates) {
      const TrackRef pred{psPar, inChg, N_proc};
      const PlaneRef pl{plPnt, plNrm};

      MPlexQF bFld;
      local_field(pred, pflags.use_param_b_field, bFld, pflags.use_fit_b_field);

      LocalPred L;
      to_local(pred, psErr, pl, plDir, bFld, L);

      LocalMeas M;
      measurement(L, msPar, msErr, plPnt, M);
      if (doCPE && cpe_corr_func)
        measurement_cpe(*doCPE, cpe_corr_func, L, M);

      Residual R;
      residual(L, M, R);
      chi2(R, outChi2);

      LocalUpd U;
      update_joseph(L, M, R, U);
      // the local states the caller asked for (a null pointer: not needed)
      if (localStates) {
        if (localStates->predPar)
          *localStates->predPar = L.lp;
        if (localStates->predErr)
          *localStates->predErr = L.err;
        if (localStates->updPar)
          *localStates->updPar = U.lp;
        if (localStates->updErr)
          *localStates->updErr = U.err;
        if (localStates->pzSign)
          *localStates->pzSign = L.pz_sign;
      }
      to_global(U, L, pl, inChg, bFld, outPar, outErr);
    }

  }  // namespace

  void propagate(const MPlexLS& inErr,
                 const MPlexLV& inPar,
                 const MPlexQI& inChg,
                 const MPlexHV& plPnt,
                 const MPlexHV& plNrm,
                 MPlexLS& outErr,
                 MPlexLV& outPar,
                 MPlexQI& outFailFlag,
                 const int N_proc,
                 const Pass& pass,
                 const MPlexQI* noMatEffPtr,
                 const ModuleMaterial* modMat) {
    const PropagationFlags& pflags = pass.pflags;
    const TrackRef in{inPar, inChg, N_proc};
    const PlaneRef pl{plPnt, plNrm};

    MPlexLL errorProp{0.0f};
    MPlexQF s{0.0f};
    outFailFlag.setVal(0.f);

    step_to_plane(in, pl, pflags, pass.ffflags, outPar, s, &errorProp);

    finish_on_plane(in, inErr, plNrm, errorProp, s, outErr, outPar, outFailFlag, pass, noMatEffPtr, modMat);
  }

  void propagate_sub_steps(const MPlexLS& inErr,
                           const MPlexLV& inPar,
                           const MPlexQI& inChg,
                           const MPlexHV& plPnt,
                           const MPlexHV& plNrm,
                           MPlexLS& outErr,
                           MPlexLV& outPar,
                           MPlexQI& outFailFlag,
                           const int N_proc,
                           const Pass& pass,
                           const int nSub,
                           const bool* split,
                           const MPlexQI* noMatEffPtr,
                           const ModuleMaterial* modMat) {
    const PropagationFlags& pflags = pass.pflags;
    const FinalFitFlags& ffflags = pass.ffflags;
    const TrackRef in{inPar, inChg, N_proc};
    const PlaneRef pl{plPnt, plNrm};
    outFailFlag.setVal(0);

    // the first nSub - 1 sub-steps, of fixed path length (none with nSub == 1)
    MPlexLV par = inPar;
    MPF sTot{0.0f};
    if (nSub > 1) {
      MPF step = first_path_estimate(in, pl, pflags);
      for (int n = 0; n < NN; ++n)
        step[n] = (n < N_proc && (split == nullptr || split[n])) ? step[n] / nSub : 0.f;
      for (int ks = 1; ks < nSub; ++ks) {
        fixed_length_sub_step(par, inChg, step, N_proc, pflags, ffflags);
        sTot = sTot + step;
      }
    }

    // the last sub-step onto the plane, parameters only
    MPF sLast{0.0f};
    step_to_plane(TrackRef{par, inChg, N_proc}, pl, pflags, ffflags, outPar, sLast, nullptr);
    sTot = sTot + sLast;

    // the field of the whole step (at the chord midpoint with b_field_at_mid), and in it the Jacobian errorProp
    // of the one-step transport from the start over the accumulated path length
    FieldAt fW;
    field_at_start(in, pflags, fW);
    if (pflags.use_param_b_field && ffflags.b_field_at_mid)
      field_at_mid(in, outPar, fW);
    StartTrig t;
    start_trig(in, t);
    MPlexLV parJ{0.0f};
    MPlexLL errorProp{0.0f};
    drift(in, t, fW, sTot, parJ);
    jacobian(in, t, parJ, fW, sTot, errorProp);

    finish_on_plane(in, inErr, plNrm, errorProp, sTot, outErr, outPar, outFailFlag, pass, noMatEffPtr, modMat);
  }

  void propagate_update(const MPlexLS& psErr,
                        const MPlexLV& psPar,
                        MPlexQI& Chg,
                        const MPlexHS& msErr,
                        const MPlexHV& msPar,
                        const MPlexHV& plNrm,
                        const MPlexHV& plDir,
                        const MPlexHV& plPnt,
                        MPlexLS& outErr,
                        MPlexLV& outPar,
                        MPlexQI& outFailFlag,
                        MPlexQF& outChi2,
                        const int N_proc,
                        const Pass& pass,
                        const MPlexQI* noMatEffPtr,
                        const MPlexQI* doCPE,
                        cpe_func cpe_corr_func,
                        const ModuleMaterial* modMat,
                        const LocalStatesOut* localStates) {
    // Sub-steps for the lanes not yet on their plane.
    bool split[NN] = {false};
    bool any_split = false;
    if (pass.n_sub > 1) {
      for (int i = 0; i < N_proc; ++i) {
        const float d = (plPnt.constAt(i, 0, 0) - psPar.constAt(i, 0, 0)) * plNrm.constAt(i, 0, 0) +
                        (plPnt.constAt(i, 1, 0) - psPar.constAt(i, 1, 0)) * plNrm.constAt(i, 1, 0) +
                        (plPnt.constAt(i, 2, 0) - psPar.constAt(i, 2, 0)) * plNrm.constAt(i, 2, 0);
        if (d != 0.f) {
          split[i] = true;
          any_split = true;
        }
      }
    }

    MPlexLS propErr;
    MPlexLV propPar;
    if (any_split)
      propagate_sub_steps(psErr,
                          psPar,
                          Chg,
                          plPnt,
                          plNrm,
                          propErr,
                          propPar,
                          outFailFlag,
                          N_proc,
                          pass,
                          pass.n_sub,
                          split,
                          noMatEffPtr,
                          modMat);
    else
      propagate(psErr, psPar, Chg, plPnt, plNrm, propErr, propPar, outFailFlag, N_proc, pass, noMatEffPtr, modMat);
    update_on_plane(propErr,
                    propPar,
                    Chg,
                    msErr,
                    msPar,
                    plNrm,
                    plDir,
                    plPnt,
                    outErr,
                    outPar,
                    outChi2,
                    N_proc,
                    pass.pflags,
                    doCPE,
                    cpe_corr_func,
                    localStates);
    for (int n = 0; n < NN; ++n) {
      if (outPar.At(n, 3, 0) < 0) {
        Chg.At(n, 0, 0) = -Chg.At(n, 0, 0);
        outPar.At(n, 3, 0) = -outPar.At(n, 3, 0);
      }
    }
  }

  //==============================================================================
  // Per-hit states of the final fit: two-filter smoother on a module plane
  //==============================================================================

  void smooth_local_states(const MPlex5V& xf,
                           const MPlex5S& cf,
                           const MPlex5V& xb,
                           const MPlex5S& cb,
                           MPlex5V& xs,
                           MPlex5S& cs,
                           MPlexQI& ok,
                           const int N_proc) {
    // S = Cf + Cb = L L^T (Cholesky, L = chol).  With Yf = L^-1 Cf (yFwd) and z = L^-1 (xb - xf) (zRes):
    //   xs = xf + Cf S^-1 (xb - xf) = xf + Yf^T z,
    //   Cs = (I - K) Cf (I - K)^T + K Cb K^T with the gain K = Cf S^-1 (Joseph form; equal to Cf S^-1 Cb).
    // Cs as a sum of two congruences, not Cf - Cf S^-1 Cf, so no cancellation when Cb << Cf, and positive
    // semi-definite also in single precision.  Every innermost loop runs over the NN lanes and vectorises; the
    // elements of the symmetric inputs are read through row pointers taken outside them.
    constexpr int D = 5;
    // Work arrays in the Matriplex layout [row][column][lane], plain: GCC vectorises the lane loops over these, not
    // over Matriplex accessors in the same loop nests.
    float chol[D][D][NN];      // L, lower triangle
    float cholInvDiag[D][NN];  // 1 / L(j, j)
    float acc[NN];             // per-lane accumulator
#pragma omp simd
    for (int n = 0; n < NN; ++n)
      ok(n, 0, 0) = 1;
    for (int j = 0; j < D; ++j) {
      const float* cfJJ = &cf.constAt(0, j, j);
      const float* cbJJ = &cb.constAt(0, j, j);
#pragma omp simd
      for (int n = 0; n < NN; ++n)
        acc[n] = cfJJ[n] + cbJJ[n];
      for (int k = 0; k < j; ++k)
#pragma omp simd
        for (int n = 0; n < NN; ++n)
          acc[n] -= chol[j][k][n] * chol[j][k][n];
#pragma omp simd
      for (int n = 0; n < NN; ++n) {
        ok(n, 0, 0) &= acc[n] > 0.f;
        acc[n] = acc[n] > 1.e-30f ? acc[n] : 1.e-30f;
      }
      // (a separate loop, so that the sqrt and the division vectorise also with trapping math)
#pragma omp simd
      for (int n = 0; n < NN; ++n) {
        chol[j][j][n] = std::sqrt(acc[n]);
        cholInvDiag[j][n] = 1.f / chol[j][j][n];
      }
      for (int i = j + 1; i < D; ++i) {
        const float* cfIJ = &cf.constAt(0, i, j);
        const float* cbIJ = &cb.constAt(0, i, j);
#pragma omp simd
        for (int n = 0; n < NN; ++n)
          acc[n] = cfIJ[n] + cbIJ[n];
        for (int k = 0; k < j; ++k)
#pragma omp simd
          for (int n = 0; n < NN; ++n)
            acc[n] -= chol[i][k][n] * chol[j][k][n];
#pragma omp simd
        for (int n = 0; n < NN; ++n)
          chol[i][j][n] = acc[n] * cholInvDiag[j][n];
      }
    }

    // forward substitution L y = b for b = xb - xf and the five columns of Cf
    float zRes[D][NN], yFwd[D][D][NN];
    for (int i = 0; i < D; ++i) {
      const float* xbI = &xb.constAt(0, i, 0);
      const float* xfI = &xf.constAt(0, i, 0);
#pragma omp simd
      for (int n = 0; n < NN; ++n)
        zRes[i][n] = xbI[n] - xfI[n];
      for (int k = 0; k < i; ++k)
#pragma omp simd
        for (int n = 0; n < NN; ++n)
          zRes[i][n] -= chol[i][k][n] * zRes[k][n];
#pragma omp simd
      for (int n = 0; n < NN; ++n)
        zRes[i][n] *= cholInvDiag[i][n];
    }
    for (int c = 0; c < D; ++c)
      for (int i = 0; i < D; ++i) {
        const float* cfIC = &cf.constAt(0, i, c);
#pragma omp simd
        for (int n = 0; n < NN; ++n)
          yFwd[i][c][n] = cfIC[n];
        for (int k = 0; k < i; ++k)
#pragma omp simd
          for (int n = 0; n < NN; ++n)
            yFwd[i][c][n] -= chol[i][k][n] * yFwd[k][c][n];
#pragma omp simd
        for (int n = 0; n < NN; ++n)
          yFwd[i][c][n] *= cholInvDiag[i][n];
      }

    for (int i = 0; i < D; ++i) {
      const float* xfI = &xf.constAt(0, i, 0);
#pragma omp simd
      for (int n = 0; n < NN; ++n)
        acc[n] = xfI[n];
      for (int k = 0; k < D; ++k)
#pragma omp simd
        for (int n = 0; n < NN; ++n)
          acc[n] += yFwd[k][i][n] * zRes[k][n];
      float* xsI = &xs.At(0, i, 0);
#pragma omp simd
      for (int n = 0; n < NN; ++n)
        xsI[n] = acc[n];
    }

    // the gain, K^T = S^-1 Cf, by back substitution L^T K^T = Yf
    float gainT[D][D][NN];
    for (int c = 0; c < D; ++c)
      for (int i = D - 1; i >= 0; --i) {
#pragma omp simd
        for (int n = 0; n < NN; ++n)
          acc[n] = yFwd[i][c][n];
        for (int k = i + 1; k < D; ++k)
#pragma omp simd
          for (int n = 0; n < NN; ++n)
            acc[n] -= chol[k][i][n] * gainT[k][c][n];
#pragma omp simd
        for (int n = 0; n < NN; ++n)
          gainT[i][c][n] = acc[n] * cholInvDiag[i][n];
      }
    // fwdTerm = (I - K) Cf and gainCb = K Cb, with K(i, m) = gainT(m, i)
    float fwdTerm[D][D][NN], gainCb[D][D][NN];
    for (int i = 0; i < D; ++i)
      for (int j = 0; j < D; ++j) {
        const float* cfIJ = &cf.constAt(0, i, j);
#pragma omp simd
        for (int n = 0; n < NN; ++n) {
          fwdTerm[i][j][n] = cfIJ[n];
          gainCb[i][j][n] = 0.f;
        }
        for (int m = 0; m < D; ++m) {
          const float* cfMJ = &cf.constAt(0, m, j);
          const float* cbMJ = &cb.constAt(0, m, j);
#pragma omp simd
          for (int n = 0; n < NN; ++n) {
            fwdTerm[i][j][n] -= gainT[m][i][n] * cfMJ[n];
            gainCb[i][j][n] += gainT[m][i][n] * cbMJ[n];
          }
        }
      }
    // Cs = fwdTerm (I - K)^T + gainCb K^T
    for (int i = 0; i < D; ++i)
      for (int j = 0; j <= i; ++j) {
#pragma omp simd
        for (int n = 0; n < NN; ++n)
          acc[n] = fwdTerm[i][j][n];
        for (int m = 0; m < D; ++m)
#pragma omp simd
          for (int n = 0; n < NN; ++n)
            acc[n] += (gainCb[i][m][n] - fwdTerm[i][m][n]) * gainT[m][j][n];
        float* csIJ = &cs.At(0, i, j);
#pragma omp simd
        for (int n = 0; n < NN; ++n)
          csIJ[n] = acc[n];
      }
  }
}  // namespace mkfit::final_fit
