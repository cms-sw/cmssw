#include "FWCore/Utilities/interface/CMSUnrollLoop.h"
#include "RecoTracker/MkFitCore/interface/PropagationConfig.h"
#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkFitCore/interface/cms_common_macros.h"

#include "PropagationMPlex.h"

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
                          MPlexLV& __restrict__ outPar,
                          const MPlexQF& __restrict__ kinv,
                          const MPlexQF& __restrict__ s) {
    namespace mpt = Matriplex;
    using MPF = MPlexQF;

    MPF alpha = s * mpt::fast_sin(inPar(5, 0)) * inPar(3, 0) * kinv;

    MPF sinah, cosah;
    if constexpr (Config::useTrigApprox) {
      mpt::sincos4(0.5f * alpha, sinah, cosah);
    } else {
      mpt::fast_sincos(0.5f * alpha, sinah, cosah);
    }

    MPF sin_mom_phi, cos_mom_phi;
    mpt::fast_sincos(inPar(4, 0), sin_mom_phi, cos_mom_phi);

    MPF sin_mom_tht, cos_mom_tht;
    mpt::fast_sincos(inPar(5, 0), sin_mom_tht, cos_mom_tht);

    outPar.aij(0, 0) = inPar(0, 0) + 2.f * sinah * (cos_mom_phi * cosah - sin_mom_phi * sinah) / (inPar(3, 0) * kinv);
    outPar.aij(1, 0) = inPar(1, 0) + 2.f * sinah * (sin_mom_phi * cosah + cos_mom_phi * sinah) / (inPar(3, 0) * kinv);
    outPar.aij(2, 0) = inPar(2, 0) + alpha / kinv * cos_mom_tht / (inPar(3, 0) * sin_mom_tht);
    outPar.aij(3, 0) = inPar(3, 0);
    outPar.aij(4, 0) = inPar(4, 0) + alpha;
    outPar.aij(5, 0) = inPar(5, 0);
  }

  //*****************************************************************************************************

  //should kinv and D be templated???
  void parsAndErrPropFromPathL_impl(const MPlexLV& __restrict__ inPar,
                                    const MPlexQI& __restrict__ inChg,
                                    MPlexLV& __restrict__ outPar,
                                    const MPlexQF& __restrict__ kinv,
                                    const MPlexQF& __restrict__ s,
                                    MPlexLL& __restrict__ errorProp,
                                    const int N_proc,
                                    const PropagationFlags& pf) {
    //iteration should return the path length s, then update parameters and compute errors

    namespace mpt = Matriplex;
    using MPF = MPlexQF;

    parsFromPathL_impl(inPar, outPar, kinv, s);

    MPF sinPin, cosPin;
    mpt::fast_sincos(inPar(4, 0), sinPin, cosPin);
    MPF sinPout, cosPout;
    mpt::fast_sincos(outPar(4, 0), sinPout, cosPout);
    MPF sinT, cosT;
    mpt::fast_sincos(inPar(5, 0), sinT, cosT);

    // use code from AnalyticalCurvilinearJacobian::computeFullJacobian for error propagation in curvilinear coordinates, then convert to CCS
    // main difference from the above function is that we assume that the magnetic field is purely along z (which also implies that there is no change in pz)
    // this simplifies significantly the code

    MPlex55 errorPropCurv;

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
    const MPF bF = (pf.use_param_b_field ? Const::sol_over_100 * getBFieldFromZXY(inPar(2, 0), inPar(0, 0), inPar(1, 0))
                                         : Const::sol_over_100 * Config::Bfield);
    const MPF q = -bF * qbp;
    const MPF theta = q * s;
    MPF sint, cost;
    mpt::fast_sincos(theta, sint, cost);
    const MPF dx1 = inPar(0, 0) - outPar(0, 0);
    const MPF dx2 = inPar(1, 0) - outPar(1, 0);
    const MPF dx3 = inPar(2, 0) - outPar(2, 0);
    MPF au = mpt::fast_isqrt(t11 * t11 + t12 * t12);
    const MPF u11 = -au * t12;
    const MPF u12 = au * t11;
    const MPF v11 = -cosT * u12;
    const MPF v12 = cosT * u11;
    const MPF v13 = t11 * u12 - t12 * u11;
    au = mpt::fast_isqrt(t21 * t21 + t22 * t22);
    const MPF u21 = -au * t22;
    const MPF u22 = au * t21;
    const MPF v21 = -cosT * u22;
    const MPF v22 = cosT * u21;
    const MPF v23 = t21 * u22 - t22 * u21;
    // now prepare the transport matrix
    const MPF omcost = 1.f - cost;
    const MPF tmsint = theta - sint;
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
      float cutCriterion = fabs(s[n] * sinT[n] * inPar(n, 3, 0));
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
        const float temp4 = -t11[n] * v21[n] - t12[n] * v22[n] - cosT[n] * v23[n];
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
             float pt,
             int q,
             float kinv) {
    float A = delta0 * eta0 + delta1 * eta1 + delta2 * eta2;
    float ip = sinT / pt;
    float p0[3] = {pt * cosP, pt * sinP, cosT / ip};
    float B = (p0[0] * eta0 + p0[1] * eta1 + p0[2] * eta2) * ip;
    float rho = kinv * ip;
    float C = (eta0 * p0[1] - eta1 * p0[0]) * rho * 0.5f * ip;
    float sqb2m4ac = std::sqrt(B * B - 4.f * A * C);
    float s1 = (-B + sqb2m4ac) * 0.5f / C;
    float s2 = (-B - sqb2m4ac) * 0.5f / C;
#ifdef DEBUG
    if (debug)
      std::cout << "A=" << A << " B=" << B << " C=" << C << " s1=" << s1 << " s2=" << s2 << std::endl;
#endif
    //take the closest
    return (std::abs(s1) > std::abs(s2) ? s2 : s1);
  }

  void helixAtPlane_impl(const MPlexLV& __restrict__ inPar,
                         const MPlexQI& __restrict__ inChg,
                         const MPlexHV& __restrict__ plPnt,
                         const MPlexHV& __restrict__ plNrm,
                         MPlexQF& __restrict__ s,
                         MPlexLV& __restrict__ outPar,
                         MPlexLL& __restrict__ errorProp,
                         MPlexQI& __restrict__ outFailFlag,  // expected to be initialized to 0
                         const int N_proc,
                         const PropagationFlags& pf) {
    namespace mpt = Matriplex;
    using MPF = MPlexQF;

#ifdef DEBUG
    for (int n = 0; n < N_proc; ++n) {
      dprint_np(n,
                "input parameters"
                    << " inPar(n, 0, 0)=" << std::setprecision(9) << inPar(n, 0, 0) << " inPar(n, 1, 0)="
                    << std::setprecision(9) << inPar(n, 1, 0) << " inPar(n, 2, 0)=" << std::setprecision(9)
                    << inPar(n, 2, 0) << " inPar(n, 3, 0)=" << std::setprecision(9) << inPar(n, 3, 0)
                    << " inPar(n, 4, 0)=" << std::setprecision(9) << inPar(n, 4, 0)
                    << " inPar(n, 5, 0)=" << std::setprecision(9) << inPar(n, 5, 0));
    }
#endif

    MPF kinv = mpt::negate_if_ltz(MPF(-Const::sol_over_100), inChg);
    if (pf.use_param_b_field) {
      kinv *= getBFieldFromZXY(inPar(2, 0), inPar(0, 0), inPar(1, 0));
    } else {
      kinv *= Config::Bfield;
    }

    MPF delta0 = inPar(0, 0) - plPnt(0, 0);
    MPF delta1 = inPar(1, 0) - plPnt(1, 0);
    MPF delta2 = inPar(2, 0) - plPnt(2, 0);

    MPF sinP, cosP;
    mpt::fast_sincos(inPar(4, 0), sinP, cosP);
    MPF sinT, cosT;
    mpt::fast_sincos(inPar(5, 0), sinT, cosT);

    // determine solution for straight line
    MPF sl = -(plNrm(0, 0) * delta0 + plNrm(1, 0) * delta1 + plNrm(2, 0) * delta2) /
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

    MPlexLV outParTmp;

    CMS_UNROLL_LOOP_COUNT(Config::Niter - 1)
    for (int i = 0; i < Config::Niter - 1; ++i) {
      parsFromPathL_impl(inPar, outParTmp, kinv, s);

      delta0 = outParTmp(0, 0) - plPnt(0, 0);
      delta1 = outParTmp(1, 0) - plPnt(1, 0);
      delta2 = outParTmp(2, 0) - plPnt(2, 0);

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
    }  //end Niter-1

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
    parsAndErrPropFromPathL_impl(inPar, inChg, outPar, kinv, s, errorProp, N_proc, pf);
  }

}  // namespace
// END STUFF FROM PropagationMPlex.icc
// ============================================================================

namespace mkfit {

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

    helixAtPlane_impl(inPar, inChg, plPnt, plNrm, pathL, outPar, errorProp, outFailFlag, N_proc, pflags);
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

    outErr = inErr;
    outPar = inPar;

    MPlexQF pathL;
    MPlexLL errorProp;

    helixAtPlane(inPar, inChg, plPnt, plNrm, pathL, outPar, errorProp, outFailFlag, N_proc, pflags);

#ifdef DEBUG
    for (int n = 0; n < N_proc; ++n) {
      dprint_np(
          n,
          "propagation to plane end, dump parameters\n"
              //<< "   D = " << s[n] << " alpha = " << s[n] * std::sin(inPar(n, 5, 0)) * inPar(n, 3, 0) * kinv[n] << " kinv = " << kinv[n] << std::endl
              << "   pos = " << outPar(n, 0, 0) << " " << outPar(n, 1, 0) << " " << outPar(n, 2, 0) << "\t\t r="
              << std::sqrt(outPar(n, 0, 0) * outPar(n, 0, 0) + outPar(n, 1, 0) * outPar(n, 1, 0)) << std::endl
              << "   mom = " << outPar(n, 3, 0) << " " << outPar(n, 4, 0) << " " << outPar(n, 5, 0) << std::endl
              << " charge = " << inChg(n, 0, 0) << std::endl
              << " cart= " << std::cos(outPar(n, 4, 0)) / outPar(n, 3, 0) << " "
              << std::sin(outPar(n, 4, 0)) / outPar(n, 3, 0) << " " << 1. / (outPar(n, 3, 0) * tan(outPar(n, 5, 0)))
              << "\t\tpT=" << 1. / std::abs(outPar(n, 3, 0)) << std::endl);
    }

    if (debug && g_debug) {
      for (int kk = 0; kk < N_proc; ++kk) {
        dprintf("inPar %d\n", kk);
        for (int i = 0; i < 6; ++i) {
          dprintf("%8f ", inPar.constAt(kk, i, 0));
        }
        dprintf("\n");
        dprintf("inErr %d\n", kk);
        for (int i = 0; i < 6; ++i) {
          for (int j = 0; j < 6; ++j)
            dprintf("%8f ", inErr.constAt(kk, i, j));
          dprintf("\n");
        }
        dprintf("\n");

        for (int kk = 0; kk < N_proc; ++kk) {
          dprintf("plNrm %d\n", kk);
          for (int j = 0; j < 3; ++j)
            dprintf("%8f ", plNrm.constAt(kk, 0, j));
        }
        dprintf("\n");

        for (int kk = 0; kk < N_proc; ++kk) {
          dprintf("pathL %d\n", kk);
          for (int j = 0; j < 1; ++j)
            dprintf("%8f ", pathL.constAt(kk, 0, j));
        }
        dprintf("\n");

        dprintf("errorProp %d\n", kk);
        for (int i = 0; i < 6; ++i) {
          for (int j = 0; j < 6; ++j)
            dprintf("%8f ", errorProp.At(kk, i, j));
          dprintf("\n");
        }
        dprintf("\n");
      }
    }
#endif

    // Matriplex version of:
    // result.errors = ROOT::Math::Similarity(errorProp, outErr);
    MPlexLL temp;
    MultHelixPlaneProp(errorProp, outErr, temp);
    MultHelixPlanePropTransp(errorProp, temp, outErr);
    // MultHelixPropFull(errorProp, outErr, temp);
    // for (int kk = 0; kk < 1; ++kk) {
    //   std::cout << "errorProp" << std::endl;
    //   for (int i = 0; i < 6; ++i) {
    // 	for (int j = 0; j < 6; ++j)
    // 	  std::cout << errorProp.constAt(kk, i, j) << " ";
    // 	std::cout << std::endl;;
    //   }
    //   std::cout << std::endl;;
    //   std::cout << "outErr" << std::endl;
    //   for (int i = 0; i < 6; ++i) {
    // 	for (int j = 0; j < 6; ++j)
    // 	  std::cout << outErr.constAt(kk, i, j) << " ";
    // 	std::cout << std::endl;;
    //   }
    //   std::cout << std::endl;;
    //   std::cout << "temp" << std::endl;
    //   for (int i = 0; i < 6; ++i) {
    // 	for (int j = 0; j < 6; ++j)
    // 	  std::cout << temp.constAt(kk, i, j) << " ";
    // 	std::cout << std::endl;;
    //   }
    //   std::cout << std::endl;;
    // }
    // MultHelixPropTranspFull(errorProp, temp, outErr);

#ifdef DEBUG
    if (debug && g_debug) {
      for (int kk = 0; kk < N_proc; ++kk) {
        dprintf("outErr %d\n", kk);
        for (int i = 0; i < 6; ++i) {
          for (int j = 0; j < 6; ++j)
            dprintf("%8f ", outErr.constAt(kk, i, j));
          dprintf("\n");
        }
        dprintf("\n");
      }
    }
#endif

    if (pflags.apply_material) {
      MPlexQF hitsRl;
      MPlexQF hitsXi;
      MPlexQF propSign;

      const TrackerInfo& tinfo = *pflags.tracker_info;

#if !defined(__clang__)
#pragma omp simd
#endif
      for (int n = 0; n < NN; ++n) {
        if (n >= N_proc || (noMatEffPtr && noMatEffPtr->constAt(n, 0, 0))) {
          hitsRl(n, 0, 0) = 0.f;
          hitsXi(n, 0, 0) = 0.f;
        } else {
          const float hypo = hipo(outPar(n, 0, 0), outPar(n, 1, 0));
          const auto mat = tinfo.material_checked(std::abs(outPar(n, 2, 0)), hypo);
          hitsRl(n, 0, 0) = mat.radl;
          hitsXi(n, 0, 0) = mat.bbxi;
        }
        propSign(n, 0, 0) = (pathL(n, 0, 0) > 0.f ? 1.f : -1.f);
      }
      applyMaterialEffects(hitsRl, hitsXi, propSign, plNrm, outErr, outPar, N_proc);
#ifdef DEBUG
      if (debug && g_debug) {
        for (int kk = 0; kk < N_proc; ++kk) {
          dprintf("propSign %d\n", kk);
          for (int i = 0; i < 1; ++i) {
            dprintf("%8f ", propSign.constAt(kk, i, 0));
          }
          dprintf("\n");
          dprintf("plNrm %d\n", kk);
          for (int i = 0; i < 3; ++i) {
            dprintf("%8f ", plNrm.constAt(kk, i, 0));
          }
          dprintf("\n");
          dprintf("outErr(after material) %d\n", kk);
          for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j)
              dprintf("%8f ", outErr.constAt(kk, i, j));
            dprintf("\n");
          }
          dprintf("\n");
        }
      }
#endif
    }

    squashPhiMPlex(outPar, N_proc);  // ensure phi is between |pi|

    // PROP-FAIL-ENABLE To keep physics changes minimal, we always restore the
    // state to input when propagation fails -- as was the default before.
    // if (pflags.copy_input_state_on_fail) {
    for (int i = 0; i < N_proc; ++i) {
      if (outFailFlag(i, 0, 0)) {
        outPar.copySlot(i, inPar);
        outErr.copySlot(i, inErr);
      }
    }
    // }
  }

}  // namespace mkfit
