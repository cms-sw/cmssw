#include "FWCore/Utilities/interface/CMSUnrollLoop.h"
#include "RecoTracker/MkFitCore/interface/PropagationConfig.h"
#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkFitCore/interface/cms_common_macros.h"

#include "PropagationMPlex.h"
#include "PlaneSteps.h"

//#define DEBUG
#include "Debug.h"

namespace {
  using namespace mkfit;

  void MultHelixPlaneProp(const MPlexLL& A, const MPlexLS& B, MPlexLL& C) {
    // C = A * B

    typedef float T;
    const Matriplex::idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#include "MultHelixPlaneProp.ah"
  }

  void MultHelixPlanePropTransp(const MPlexLL& A, const MPlexLL& B, MPlexLS& C) {
    // C = B * AT;

    typedef float T;
    const Matriplex::idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#include "MultHelixPlanePropTransp.ah"
  }

}  // namespace

// ============================================================================
// BEGIN STUFF FROM PropagationMPlex.icc
namespace {

  using MPF = MPlexQF;

  MPF getBFieldFromZXY(const MPF& z, const MPF& x, const MPF& y) {
    MPF b;
    for (int n = 0; n < NN; ++n)
      b[n] = Config::bFieldFromZR(z[n], hipo(x[n], y[n]));
    return b;
  }

  void JacErrPropCurv1(const MPlex65& A, const MPlex55& B, MPlex65& C) {
    // C = A * B
    typedef float T;
    const Matriplex::idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#include "JacErrPropCurv1.ah"
  }

  void JacErrPropCurv2(const MPlex65& A, const MPlex56& B, MPlexLL& __restrict__ C) {
    // C = A * B
    typedef float T;
    const Matriplex::idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#include "JacErrPropCurv2.ah"
  }

  void parsFromPathL_impl(const MPlexLV& __restrict__ inPar,
                          const MPlexQF& __restrict__ sin_mom_phi,
                          const MPlexQF& __restrict__ cos_mom_phi,
                          const MPlexQF& __restrict__ sin_mom_tht,
                          const MPlexQF& __restrict__ cos_mom_tht,
                          MPlexLV& __restrict__ outPar,
                          const MPlexQF& __restrict__ kinv,
                          const MPlexQF& __restrict__ s) {
    namespace mpt = Matriplex;
    using MPF = MPlexQF;

    const MPF alpha = s * sin_mom_tht * inPar(3, 0) * kinv;

    MPF sinah, cosah;
    if constexpr (Config::useTrigApprox) {
      mpt::sincos4(0.5f * alpha, sinah, cosah);
    } else {
      mpt::fast_sincos(0.5f * alpha, sinah, cosah);
    }

    outPar.aij(0, 0) = inPar(0, 0) + 2.f * sinah * (cos_mom_phi * cosah - sin_mom_phi * sinah) / (inPar(3, 0) * kinv);
    outPar.aij(1, 0) = inPar(1, 0) + 2.f * sinah * (sin_mom_phi * cosah + cos_mom_phi * sinah) / (inPar(3, 0) * kinv);
    outPar.aij(2, 0) = inPar(2, 0) + alpha / kinv * cos_mom_tht / (inPar(3, 0) * sin_mom_tht);
    outPar.aij(3, 0) = inPar(3, 0);
    outPar.aij(4, 0) = inPar(4, 0) + alpha;
    outPar.aij(5, 0) = inPar(5, 0);
  }

  //*****************************************************************************************************

  // Transport Jacobian of a helix step of path length s from inPar to outPar, in the field bFld [T]
  // (Bz along z, the field the step was taken in).
  void errPropFromPathL_impl(const MPlexLV& __restrict__ inPar,
                             const MPlexQI& __restrict__ inChg,
                             const MPlexQF& __restrict__ sinPin,
                             const MPlexQF& __restrict__ cosPin,
                             const MPlexQF& __restrict__ sinT,
                             const MPlexQF& __restrict__ cosT,
                             const MPlexLV& __restrict__ outPar,
                             const MPlexQF& __restrict__ bFld,
                             const MPlexQF& __restrict__ s,
                             MPlexLL& __restrict__ errorProp,
                             const int N_proc) {
    namespace mpt = Matriplex;
    using MPF = MPlexQF;

    MPF sinPout, cosPout;
    mpt::fast_sincos(outPar(4, 0), sinPout, cosPout);

    // use code from AnalyticalCurvilinearJacobian::computeFullJacobian for error propagation in curvilinear coordinates, then convert to CCS
    // main difference from the above function is that we assume that the magnetic field is purely along z (which also implies that there is no change in pz)
    // this simplifies significantly the code
    const MPF qbp = mpt::negate_if_ltz(sinT * inPar(3, 0), inChg);
    // calculate transport matrix
    // Origin: TRPRFN
    const MPF t11 = cosPin * sinT;
    const MPF t12 = sinPin * sinT;
    const MPF t21 = cosPout * sinT;
    const MPF t22 = sinPout * sinT;
    const MPF cosl1 = 1.f / sinT;
    // define average magnetic field and gradient
    // at initial point - inlike TRPRFN
    const MPF bF = Const::sol_over_100 * bFld;
    const MPF q = -bF * qbp;
    const MPF theta = q * s;
    MPF sint, cost;
    mpt::fast_sincos(theta, sint, cost);
    const MPF dx1 = inPar(0, 0) - outPar(0, 0);
    const MPF dx2 = inPar(1, 0) - outPar(1, 0);
    const MPF dx3 = inPar(2, 0) - outPar(2, 0);
    const MPF u11 = -sinPin;
    const MPF u12 = cosPin;
    const MPF v11 = -cosT * u12;
    const MPF v12 = cosT * u11;
    const MPF v13 = sinT;
    const MPF u21 = -sinPout;
    const MPF u22 = cosPout;
    const MPF v21 = -cosT * u22;
    const MPF v22 = cosT * u21;
    const MPF v23 = sinT;
    // now prepare the transport matrix
    const MPF omcost = 1.f - cost;
    const MPF tmsint = theta - sint;

    MPlex55 errorPropCurv{0.0f};
    //   1/p - doesn't change since |p1| = |p2|
    errorPropCurv.aij(0, 0) = 1.f;
    for (int i = 1; i < 5; ++i)
      errorPropCurv.aij(0, i) = 0.f;
    //   lambda
    errorPropCurv.aij(1, 0) = 0.f;
    errorPropCurv.aij(1, 1) =
        cost * (v11 * v21 + v12 * v22 + v13 * v23) + sint * (-v12 * v21 + v11 * v22) + omcost * v13 * v23;
    errorPropCurv.aij(1, 2) = (cost * (u11 * v21 + u12 * v22) + sint * (-u12 * v21 + u11 * v22)) * sinT;
    errorPropCurv.aij(1, 3) = 0.f;
    errorPropCurv.aij(1, 4) = 0.f;
    //   phi
    errorPropCurv.aij(2, 0) = bF * v23 * (t21 * dx1 + t22 * dx2 + cosT * dx3) * cosl1;
    errorPropCurv.aij(2, 1) = (cost * (v11 * u21 + v12 * u22) + sint * (-v12 * u21 + v11 * u22) +
                               v23 * (-sint * (v11 * t21 + v12 * t22 + v13 * cosT) + omcost * (-v11 * t22 + v12 * t21) -
                                      tmsint * cosT * v13)) *
                              cosl1;
    errorPropCurv.aij(2, 2) = (cost * (u11 * u21 + u12 * u22) + sint * (-u12 * u21 + u11 * u22) +
                               v23 * (-sint * (u11 * t21 + u12 * t22) + omcost * (-u11 * t22 + u12 * t21))) *
                              cosl1 * sinT;
    errorPropCurv.aij(2, 3) = -q * v23 * (u11 * t21 + u12 * t22) * cosl1;
    errorPropCurv.aij(2, 4) = -q * v23 * (v11 * t21 + v12 * t22 + v13 * cosT) * cosl1;

    //   yt
    for (int n = 0; n < N_proc; ++n) {
      const float cutCriterion = std::abs(s[n] * sinT[n] * inPar(n, 3, 0));
      const float limit = 5.f;  // valid for propagations with effectively float precision
      if (cutCriterion > limit) {
        const float pp = 1.f / qbp[n];
        errorPropCurv(n, 3, 0) = pp * (u21[n] * dx1[n] + u22[n] * dx2[n]);
        errorPropCurv(n, 4, 0) = pp * (v21[n] * dx1[n] + v22[n] * dx2[n] + v23[n] * dx3[n]);
      } else {
        const float temp1 = -t12[n] * u21[n] + t11[n] * u22[n];
        const float s2 = s[n] * s[n];
        const float secondOrder41 = -0.5f * bF[n] * temp1 * s2;
        const float temp2 = -t11[n] * u21[n] - t12[n] * u22[n];
        const float s3 = s2 * s[n];
        const float s4 = s3 * s[n];
        const float h2 = bF[n] * bF[n];
        const float h3 = h2 * bF[n];
        const float qbp2 = qbp[n] * qbp[n];
        const float thirdOrder41 = 1.f / 3 * h2 * s3 * qbp[n] * temp2;
        const float fourthOrder41 = 1.f / 8 * h3 * s4 * qbp2 * temp1;
        errorPropCurv(n, 3, 0) = secondOrder41 + (thirdOrder41 + fourthOrder41);
        const float temp3 = -t12[n] * v21[n] + t11[n] * v22[n];
        const float secondOrder51 = -0.5f * bF[n] * temp3 * s2;
        const float temp4 = -t11[n] * v21[n] - t12[n] * v22[n];
        const float thirdOrder51 = 1.f / 3 * h2 * s3 * qbp[n] * temp4;
        const float fourthOrder51 = 1.f / 8 * h3 * s4 * qbp2 * temp3;
        errorPropCurv(n, 4, 0) = secondOrder51 + (thirdOrder51 + fourthOrder51);
      }
    }

    errorPropCurv.aij(3, 1) = (sint * (v11 * u21 + v12 * u22) + omcost * (-v12 * u21 + v11 * u22)) / q;
    errorPropCurv.aij(3, 2) = (sint * (u11 * u21 + u12 * u22) + omcost * (-u12 * u21 + u11 * u22)) * sinT / q;
    errorPropCurv.aij(3, 3) = (u11 * u21 + u12 * u22);
    errorPropCurv.aij(3, 4) = (v11 * u21 + v12 * u22);
    //   zt
    errorPropCurv.aij(4, 1) =
        (sint * (v11 * v21 + v12 * v22 + v13 * v23) + omcost * (-v12 * v21 + v11 * v22) + tmsint * v23 * v13) / q;
    errorPropCurv.aij(4, 2) = (sint * (u11 * v21 + u12 * v22) + omcost * (-u12 * v21 + u11 * v22)) * sinT / q;
    errorPropCurv.aij(4, 3) = (u11 * v21 + u12 * v22);
    errorPropCurv.aij(4, 4) = (v11 * v21 + v12 * v22 + v13 * v23);

//debug = true;
#ifdef DEBUG
    for (int n = 0; n < NN; ++n) {
      if (debug && g_debug && n < N_proc) {
        dmutex_guard;
        std::cout << n << ": errorPropCurv" << std::endl;
        printf("%5f %5f %5f %5f %5f\n",
               errorPropCurv(n, 0, 0),
               errorPropCurv(n, 0, 1),
               errorPropCurv(n, 0, 2),
               errorPropCurv(n, 0, 3),
               errorPropCurv(n, 0, 4));
        printf("%5f %5f %5f %5f %5f\n",
               errorPropCurv(n, 1, 0),
               errorPropCurv(n, 1, 1),
               errorPropCurv(n, 1, 2),
               errorPropCurv(n, 1, 3),
               errorPropCurv(n, 1, 4));
        printf("%5f %5f %5f %5f %5f\n",
               errorPropCurv(n, 2, 0),
               errorPropCurv(n, 2, 1),
               errorPropCurv(n, 2, 2),
               errorPropCurv(n, 2, 3),
               errorPropCurv(n, 2, 4));
        printf("%5f %5f %5f %5f %5f\n",
               errorPropCurv(n, 3, 0),
               errorPropCurv(n, 3, 1),
               errorPropCurv(n, 3, 2),
               errorPropCurv(n, 3, 3),
               errorPropCurv(n, 3, 4));
        printf("%5f %5f %5f %5f %5f\n",
               errorPropCurv(n, 4, 0),
               errorPropCurv(n, 4, 1),
               errorPropCurv(n, 4, 2),
               errorPropCurv(n, 4, 3),
               errorPropCurv(n, 4, 4));
        printf("\n");
      }
    }
#endif

    //now we need jacobians to convert to/from curvilinear and CCS
    // code from TrackState::jacobianCCSToCurvilinear
    MPlex56 jacCCS2Curv(0.0f);
    jacCCS2Curv.aij(0, 3) = mpt::negate_if_ltz(sinT, inChg);
    jacCCS2Curv.aij(0, 5) = mpt::negate_if_ltz(cosT * inPar(3, 0), inChg);
    jacCCS2Curv.aij(1, 5) = -1.f;
    jacCCS2Curv.aij(2, 4) = 1.f;
    jacCCS2Curv.aij(3, 0) = -sinPin;
    jacCCS2Curv.aij(3, 1) = cosPin;
    jacCCS2Curv.aij(4, 0) = -cosPin * cosT;
    jacCCS2Curv.aij(4, 1) = -sinPin * cosT;
    jacCCS2Curv.aij(4, 2) = sinT;

    // code from TrackState::jacobianCurvilinearToCCS
    MPlex65 jacCurv2CCS(0.0f);
    jacCurv2CCS.aij(0, 3) = -sinPout;
    jacCurv2CCS.aij(0, 4) = -cosT * cosPout;
    jacCurv2CCS.aij(1, 3) = cosPout;
    jacCurv2CCS.aij(1, 4) = -cosT * sinPout;
    jacCurv2CCS.aij(2, 4) = sinT;
    jacCurv2CCS.aij(3, 0) = mpt::negate_if_ltz(1.f / sinT, inChg);
    jacCurv2CCS.aij(3, 1) = outPar(3, 0) * cosT / sinT;
    jacCurv2CCS.aij(4, 2) = 1.f;
    jacCurv2CCS.aij(5, 1) = -1.f;

    //need to compute errorProp = jacCurv2CCS*errorPropCurv*jacCCS2Curv
    MPlex65 tmp;
    JacErrPropCurv1(jacCurv2CCS, errorPropCurv, tmp);
    JacErrPropCurv2(tmp, jacCCS2Curv, errorProp);
    /*
    Matriplex::multiplyGeneral(jacCurv2CCS, errorPropCurv, tmp);
    for (int kk = 0; kk < 1; ++kk) {
      std::cout << "jacCurv2CCS" << std::endl;
      for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 5; ++j)
        std::cout << jacCurv2CCS.constAt(kk, i, j) << " ";
      std::cout << std::endl;;
      }
      std::cout << std::endl;;
      std::cout << "errorPropCurv" << std::endl;
      for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 5; ++j)
        std::cout << errorPropCurv.constAt(kk, i, j) << " ";
      std::cout << std::endl;;
      }
      std::cout << std::endl;;
      std::cout << "tmp" << std::endl;
      for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 5; ++j)
        std::cout << tmp.constAt(kk, i, j) << " ";
      std::cout << std::endl;;
      }
      std::cout << std::endl;;
      std::cout << "jacCCS2Curv" << std::endl;
      for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 6; ++j)
        std::cout << jacCCS2Curv.constAt(kk, i, j) << " ";
      std::cout << std::endl;;
      }
    }
    Matriplex::multiplyGeneral(tmp, jacCCS2Curv, errorProp);
    */
  }

  // from P.Avery's notes (http://www.phys.ufl.edu/~avery/fitting/transport.pdf eq. 5)
  float getS(float delta0,
             float delta1,
             float delta2,
             float eta0,
             float eta1,
             float eta2,
             float sinP,
             float cosP,
             float sinT,
             float cosT,
             float ipt,
             int q,
             float kinv) {
    const float A = delta0 * eta0 + delta1 * eta1 + delta2 * eta2;
    const float p0[3] = {cosP * sinT, sinP * sinT, cosT};
    const float B = (p0[0] * eta0 + p0[1] * eta1 + p0[2] * eta2);
    const float rho = kinv * sinT * ipt;
    const float C = -(eta0 * p0[1] - eta1 * p0[0]) * rho * 0.5f;
    const float s1 = 2.f * A / (-B - std::copysign(std::sqrt(B * B - 4.f * A * C), B));
#ifdef DEBUG
    if (debug)
      std::cout << "A=" << A << " B=" << B << " C=" << C << " s1=" << s1 << std::endl;
#endif
    return s1;
  }

  // The path length to the plane (plPnt, plNrm) is solved in three parts: a first solve from the start
  // state, refinements from the state the current solution reaches, and the straight line where the
  // helix solution is not finite.  sinP, cosP, sinT and cosT are the start state's trigonometry; s_line
  // is its straight-line solution.

  void pathInit_impl(const MPlexLV& __restrict__ inPar,
                     const MPlexQI& __restrict__ inChg,
                     const MPlexHV& __restrict__ plPnt,
                     const MPlexHV& __restrict__ plNrm,
                     const MPlexQF& __restrict__ sinP,
                     const MPlexQF& __restrict__ cosP,
                     const MPlexQF& __restrict__ sinT,
                     const MPlexQF& __restrict__ cosT,
                     const MPlexQF& __restrict__ kinv,
                     MPlexQF& __restrict__ s,
                     MPlexQF& __restrict__ sl,
                     const int N_proc) {
    namespace mpt = Matriplex;
    using MPF = MPlexQF;

#ifdef DEBUG
    for (int n = 0; n < N_proc; ++n) {
      dprint_np(n,
                "input parameters" << " inPar(n, 0, 0)=" << std::setprecision(9) << inPar(n, 0, 0)
                                   << " inPar(n, 1, 0)=" << std::setprecision(9) << inPar(n, 1, 0)
                                   << " inPar(n, 2, 0)=" << std::setprecision(9) << inPar(n, 2, 0)
                                   << " inPar(n, 3, 0)=" << std::setprecision(9) << inPar(n, 3, 0)
                                   << " inPar(n, 4, 0)=" << std::setprecision(9) << inPar(n, 4, 0)
                                   << " inPar(n, 5, 0)=" << std::setprecision(9) << inPar(n, 5, 0));
    }
#endif

    MPF delta0 = inPar(0, 0) - plPnt(0, 0);
    MPF delta1 = inPar(1, 0) - plPnt(1, 0);
    MPF delta2 = inPar(2, 0) - plPnt(2, 0);

    // determine solution for straight line
    sl = -(plNrm(0, 0) * delta0 + plNrm(1, 0) * delta1 + plNrm(2, 0) * delta2) /
         (plNrm(0, 0) * cosP * sinT + plNrm(1, 0) * sinP * sinT + plNrm(2, 0) * cosT);

    //float s[nmax - nmin];
    //first iteration outside the loop
#pragma omp simd
    for (int n = 0; n < N_proc; ++n) {
      s[n] = (std::abs(plNrm(n, 2, 0)) < 1.f ? getS(delta0[n],
                                                    delta1[n],
                                                    delta2[n],
                                                    plNrm(n, 0, 0),
                                                    plNrm(n, 1, 0),
                                                    plNrm(n, 2, 0),
                                                    sinP[n],
                                                    cosP[n],
                                                    sinT[n],
                                                    cosT[n],
                                                    inPar(n, 3, 0),
                                                    inChg(n, 0, 0),
                                                    kinv[n])
                                             : (plPnt.constAt(n, 2, 0) - inPar.constAt(n, 2, 0)) / cosT[n]);
    }
  }

  void pathRefine_impl(const MPlexLV& __restrict__ inPar,
                       const MPlexQI& __restrict__ inChg,
                       const MPlexHV& __restrict__ plPnt,
                       const MPlexHV& __restrict__ plNrm,
                       const MPlexQF& __restrict__ sinP0,
                       const MPlexQF& __restrict__ cosP0,
                       const MPlexQF& __restrict__ sinT,
                       const MPlexQF& __restrict__ cosT,
                       const MPlexQF& __restrict__ kinv,
                       MPlexQF& __restrict__ s,
                       const int N_proc) {
    namespace mpt = Matriplex;
    using MPF = MPlexQF;

    MPlexLV outParTmp{0.0f};
    parsFromPathL_impl(inPar, sinP0, cosP0, sinT, cosT, outParTmp, kinv, s);

    const MPF delta0 = outParTmp(0, 0) - plPnt(0, 0);
    const MPF delta1 = outParTmp(1, 0) - plPnt(1, 0);
    const MPF delta2 = outParTmp(2, 0) - plPnt(2, 0);

    MPF sinP, cosP;
    mpt::fast_sincos(outParTmp(4, 0), sinP, cosP);
    // Note, sinT/cosT not updated

#pragma omp simd
    for (int n = 0; n < N_proc; ++n) {
      s[n] += (std::abs(plNrm(n, 2, 0)) < 1.f
                   ? getS(delta0[n],
                          delta1[n],
                          delta2[n],
                          plNrm(n, 0, 0),
                          plNrm(n, 1, 0),
                          plNrm(n, 2, 0),
                          sinP[n],
                          cosP[n],
                          sinT[n],
                          cosT[n],
                          inPar(n, 3, 0),
                          inChg(n, 0, 0),
                          kinv[n])
                   : (plPnt.constAt(n, 2, 0) - outParTmp.constAt(n, 2, 0)) / std::cos(outParTmp.constAt(n, 5, 0)));
    }
  }

  void pathClose_impl(const MPlexQF& __restrict__ sl, MPlexQF& __restrict__ s, const int N_proc) {
    // use linear approximation if s did not converge (for very high pT tracks)
    for (int n = 0; n < N_proc; ++n) {
#ifdef DEBUG
      if (debug)
        std::cout << "s[n]=" << s[n] << " sl[n]=" << sl[n] << " std::isnan(s[n])=" << std::isnan(s[n])
                  << " std::isfinite(s[n])=" << std::isfinite(s[n]) << " std::isnormal(s[n])=" << std::isnormal(s[n])
                  << std::endl;
#endif
      if (mkfit::isFinite(s[n]) == false && mkfit::isFinite(sl[n]))  // replace with sl even if not fully correct
        s[n] = sl[n];
    }

#ifdef DEBUG
    if (debug)
      std::cout << "s=" << s[0] << std::endl;
#endif
  }

}  // namespace
// END STUFF FROM PropagationMPlex.icc
// ============================================================================

// ============================================================================
// Steps of a propagation to a plane (see PlaneSteps.h)
// ============================================================================
namespace mkfit::plane {

  void field_at_start(const TrackRef& in, const PropagationFlags& pf, FieldAt& f) {
    f.kinv = Matriplex::negate_if_ltz(MPlexQF(-Const::sol_over_100), in.chg);
    if (pf.use_param_b_field) {
      f.b = getBFieldFromZXY(in.par(2, 0), in.par(0, 0), in.par(1, 0));
    } else {
      f.b.setVal(Config::Bfield);
    }
    f.kinv *= f.b;
  }

  void start_trig(const TrackRef& in, StartTrig& t) {
    Matriplex::fast_sincos(in.par(4, 0), t.sinP, t.cosP);
    Matriplex::fast_sincos(in.par(5, 0), t.sinT, t.cosT);
  }

  void path_init(const TrackRef& in, const StartTrig& t, const PlaneRef& pl, const FieldAt& f, PathSolve& p) {
    pathInit_impl(in.par, in.chg, pl.pnt, pl.nrm, t.sinP, t.cosP, t.sinT, t.cosT, f.kinv, p.s, p.s_line, in.n_proc);
  }

  void path_refine(const TrackRef& in, const StartTrig& t, const PlaneRef& pl, const FieldAt& f, PathSolve& p) {
    pathRefine_impl(in.par, in.chg, pl.pnt, pl.nrm, t.sinP, t.cosP, t.sinT, t.cosT, f.kinv, p.s, in.n_proc);
  }

  void path_close(const TrackRef& in, PathSolve& p) { pathClose_impl(p.s_line, p.s, in.n_proc); }

  void path_solve(const TrackRef& in, const StartTrig& t, const PlaneRef& pl, const FieldAt& f, PathSolve& p) {
    path_init(in, t, pl, f, p);
    CMS_UNROLL_LOOP_COUNT(Config::nSStepsInProp2Plane - 1)
    for (int i = 0; i < Config::nSStepsInProp2Plane - 1; ++i)
      path_refine(in, t, pl, f, p);
    path_close(in, p);
  }

  void path_from_perp(const TrackRef& in, const StartTrig& t, const MPlexQF& sPerp, MPlexQF& s) {
    s = sPerp / t.sinT;
  }

  void drift(const TrackRef& in, const StartTrig& t, const FieldAt& f, const MPlexQF& s, MPlexLV& outPar) {
    parsFromPathL_impl(in.par, t.sinP, t.cosP, t.sinT, t.cosT, outPar, f.kinv, s);
  }

  void jacobian(
      const TrackRef& in, const StartTrig& t, const MPlexLV& outPar, const FieldAt& f, const MPlexQF& s, MPlexLL& J) {
    errPropFromPathL_impl(in.par, in.chg, t.sinP, t.cosP, t.sinT, t.cosT, outPar, f.b, s, J, in.n_proc);
  }

  void transport_cov(const MPlexLL& J, const MPlexLS& inErr, MPlexLS& outErr) {
    // Matriplex version of:
    // result.errors = ROOT::Math::Similarity(errorProp, outErr);
    MPlexLL temp{0.0f};
    MultHelixPlaneProp(J, inErr, temp);
    MultHelixPlanePropTransp(J, temp, outErr);
  }

  void material_grid(
      const TrackerInfo& tinfo, const MPlexLV& par, const MPlexQI* noMatEffPtr, const int N_proc, MaterialAt& m) {
#if !defined(__clang__)
#pragma omp simd
#endif
    for (int n = 0; n < NN; ++n) {
      if (n >= N_proc || (noMatEffPtr && noMatEffPtr->constAt(n, 0, 0))) {
        m.radl(n, 0, 0) = 0.f;
        m.xi(n, 0, 0) = 0.f;
      } else {
        const float hypo = hipo(par(n, 0, 0), par(n, 1, 0));
        const auto mat = tinfo.material_checked(std::abs(par(n, 2, 0)), hypo);
        m.radl(n, 0, 0) = mat.radl;
        m.xi(n, 0, 0) = mat.bbxi;
      }
    }
  }

  void material_given(
      const MPlexQF& radl, const MPlexQF& xi, const MPlexQI* noMatEffPtr, const int N_proc, MaterialAt& m) {
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      const bool none = n >= N_proc || (noMatEffPtr && noMatEffPtr->constAt(n, 0, 0));
      m.radl(n, 0, 0) = none ? 0.f : radl(n, 0, 0);
      m.xi(n, 0, 0) = none ? 0.f : xi(n, 0, 0);
    }
  }

  void eloss_sign_from_path(const MPlexQF& s, const MPlexQI* noMatEffPtr, const int N_proc, MaterialAt& m) {
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      const bool none = n >= N_proc || (noMatEffPtr && noMatEffPtr->constAt(n, 0, 0));
      m.sign(n, 0, 0) = none ? -1.f : (s(n, 0, 0) > 0.f ? 1.f : -1.f);
    }
  }

  void eloss_sign_of_pass(const bool outward, const MPlexQI* noMatEffPtr, const int N_proc, MaterialAt& m) {
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      const bool none = n >= N_proc || (noMatEffPtr && noMatEffPtr->constAt(n, 0, 0));
      m.sign(n, 0, 0) = none ? -1.f : (outward ? 1.f : -1.f);
    }
  }

  void apply_material(const MaterialAt& m, const MPlexHV& plNrm, MPlexLS& err, MPlexLV& par, const int N_proc) {
    applyMaterialEffects(m.radl, m.xi, m.sign, plNrm, err, par, N_proc);
  }

  void finish(const TrackRef& in, const MPlexLS& inErr, const MPlexQI& failFlag, MPlexLV& par, MPlexLS& err) {
    squashPhiMPlex(par, in.n_proc);  // ensure phi is between |pi|

    // PROP-FAIL-ENABLE To keep physics changes minimal, we always restore the
    // state to input when propagation fails -- as was the default before.
    // if (pflags.copy_input_state_on_fail) {
    for (int i = 0; i < in.n_proc; ++i) {
      if (failFlag(i, 0, 0)) {
        par.copySlot(i, in.par);
        err.copySlot(i, inErr);
      }
    }
    // }
  }

}  // namespace mkfit::plane

namespace mkfit {
  using namespace plane;

  void helixAtPlane(const MPlexLV& inPar,
                    const MPlexQI& inChg,
                    const MPlexHV& plPnt,
                    const MPlexHV& plNrm,
                    MPlexQF& pathL,
                    MPlexLV& outPar,
                    MPlexLL& errorProp,
                    MPlexQI& outFailFlag,
                    const int N_proc,
                    const PropagationFlags& pflags) {
    errorProp.setVal(0.f);
    outFailFlag.setVal(0.f);

    const TrackRef in{inPar, inChg, N_proc};
    const PlaneRef pl{plPnt, plNrm};

    FieldAt f;
    field_at_start(in, pflags, f);
    StartTrig t;
    start_trig(in, t);
    PathSolve p;
    path_solve(in, t, pl, f, p);
    drift(in, t, f, p.s, outPar);
    jacobian(in, t, outPar, f, p.s, errorProp);
    for (int n = 0; n < N_proc; ++n)
      pathL[n] = p.s[n];
  }

  void propagateHelixToPlaneMPlex(const MPlexLS& inErr,
                                  const MPlexLV& inPar,
                                  const MPlexQI& inChg,
                                  const MPlexHV& plPnt,
                                  const MPlexHV& plNrm,
                                  MPlexLS& outErr,
                                  MPlexLV& outPar,
                                  MPlexQI& outFailFlag,
                                  const int N_proc,
                                  const PropagationFlags& pflags,
                                  const MPlexQI* noMatEffPtr) {
    // debug = true;

    const TrackRef in{inPar, inChg, N_proc};
    const PlaneRef pl{plPnt, plNrm};

    MPlexLL errorProp{0.0f};
    outFailFlag.setVal(0.f);

    FieldAt f;
    field_at_start(in, pflags, f);
    StartTrig t;
    start_trig(in, t);
    PathSolve p;
    path_solve(in, t, pl, f, p);
    drift(in, t, f, p.s, outPar);
    jacobian(in, t, outPar, f, p.s, errorProp);

#ifdef DEBUG
    for (int n = 0; n < N_proc; ++n) {
      dprint_np(n,
                "propagation to plane end, dump parameters\n"
                    << "   pos = " << outPar(n, 0, 0) << " " << outPar(n, 1, 0) << " " << outPar(n, 2, 0) << "\t\t r="
                    << std::sqrt(outPar(n, 0, 0) * outPar(n, 0, 0) + outPar(n, 1, 0) * outPar(n, 1, 0)) << std::endl
                    << "   mom = " << outPar(n, 3, 0) << " " << outPar(n, 4, 0) << " " << outPar(n, 5, 0) << std::endl
                    << " charge = " << inChg(n, 0, 0) << std::endl
                    << " cart= " << std::cos(outPar(n, 4, 0)) / outPar(n, 3, 0) << " "
                    << std::sin(outPar(n, 4, 0)) / outPar(n, 3, 0) << " "
                    << 1. / (outPar(n, 3, 0) * tan(outPar(n, 5, 0))) << "\t\tpT=" << 1. / std::abs(outPar(n, 3, 0))
                    << std::endl);
    }
#endif

    transport_cov(errorProp, inErr, outErr);

    if (pflags.apply_material) {
      MaterialAt m;
      material_grid(*pflags.tracker_info, outPar, noMatEffPtr, N_proc, m);
      eloss_sign_from_path(p.s, noMatEffPtr, N_proc, m);
      apply_material(m, plNrm, outErr, outPar, N_proc);
    }

    finish(in, inErr, outFailFlag, outPar, outErr);
  }

}  // namespace mkfit
