#include "FWCore/Utilities/interface/CMSUnrollLoop.h"
#include "RecoTracker/MkFitCore/interface/PropagationConfig.h"
#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"

#include "PropagationMPlex.h"

//#define DEBUG
#include "Debug.h"

namespace {
  using namespace mkfit;

  void MultHelixPropEndcap(const MPlexLL& A, const MPlexLS& B, MPlexLL& C) {
    // C = A * B

    typedef float T;
    const Matriplex::idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#include "MultHelixPropEndcap.ah"
  }

  void MultHelixPropTranspEndcap(const MPlexLL& A, const MPlexLL& B, MPlexLS& C) {
    // C = B * AT;

    typedef float T;
    const Matriplex::idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#include "MultHelixPropTranspEndcap.ah"
  }

}  // namespace

// ============================================================================
// BEGIN STUFF FROM PropagationMPlex.icc
namespace {}

// END STUFF FROM PropagationMPlex.icc
// ============================================================================

namespace mkfit {

  void propagateHelixToZMPlex(const MPlexLS& inErr,
                              const MPlexLV& inPar,
                              const MPlexQI& inChg,
                              const MPlexQF& msZ,
                              MPlexLS& outErr,
                              MPlexLV& outPar,
                              MPlexQI& outFailFlag,
                              const int N_proc,
                              const PropagationFlags& pflags,
                              const MPlexQI* noMatEffPtr) {
    // debug = true;

    outErr = inErr;
    outPar = inPar;

    MPlexLL errorProp;

    //helixAtZ_new(inPar, inChg, msZ, outPar, errorProp, outFailFlag, N_proc, pflags);
    helixAtZ(inPar, inChg, msZ, outPar, errorProp, outFailFlag, N_proc, pflags);

#ifdef DEBUG
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

    // Matriplex version of: result.errors = ROOT::Math::Similarity(errorProp, outErr);
    MPlexLL temp;
    MultHelixPropEndcap(errorProp, outErr, temp);
    MultHelixPropTranspEndcap(errorProp, temp, outErr);
    // can replace with: MultHelixPropFull(errorProp, outErr, temp); MultHelixPropTranspFull(errorProp, temp, outErr);

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
          const auto mat = tinfo.material_checked(std::abs(msZ(n, 0, 0)), hypo);
          hitsRl(n, 0, 0) = mat.radl;
          hitsXi(n, 0, 0) = mat.bbxi;
        }
        if (n < N_proc) {
          const float zout = msZ.constAt(n, 0, 0);
          const float zin = inPar.constAt(n, 2, 0);
          propSign(n, 0, 0) = (std::abs(zout) > std::abs(zin) ? 1.f : -1.f);
        }
      }
      MPlexHV plNrm;
#pragma omp simd
      for (int n = 0; n < NN; ++n) {
        plNrm(n, 0, 0) = 0.f;
        plNrm(n, 1, 0) = 0.f;
        plNrm(n, 2, 0) = 1.f;
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

  void helixAtZ(const MPlexLV& inPar,
                const MPlexQI& inChg,
                const MPlexQF& msZ,
                MPlexLV& outPar,
                MPlexLL& errorProp,
                MPlexQI& outFailFlag,
                const int N_proc,
                const PropagationFlags& pflags) {
    errorProp.setVal(0.f);
    outFailFlag.setVal(0.f);

    // debug = true;
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      //initialize erroProp to identity matrix, except element 2,2 which is zero
      errorProp(n, 0, 0) = 1.f;
      errorProp(n, 1, 1) = 1.f;
      errorProp(n, 3, 3) = 1.f;
      errorProp(n, 4, 4) = 1.f;
      errorProp(n, 5, 5) = 1.f;
    }
    float zout[NN];
    float zin[NN];
    float ipt[NN];
    float phiin[NN];
    float theta[NN];
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      //initialize erroProp to identity matrix, except element 2,2 which is zero
      zout[n] = msZ.constAt(n, 0, 0);
      zin[n] = inPar.constAt(n, 2, 0);
      ipt[n] = inPar.constAt(n, 3, 0);
      phiin[n] = inPar.constAt(n, 4, 0);
      theta[n] = inPar.constAt(n, 5, 0);
    }

    float k[NN];
    if (pflags.use_param_b_field) {
#pragma omp simd
      for (int n = 0; n < NN; ++n) {
        k[n] = inChg.constAt(n, 0, 0) * 100.f /
               (-Const::sol * Config::bFieldFromZR(zin[n], hipo(inPar.constAt(n, 0, 0), inPar.constAt(n, 1, 0))));
      }
    } else {
#pragma omp simd
      for (int n = 0; n < NN; ++n) {
        k[n] = inChg.constAt(n, 0, 0) * 100.f / (-Const::sol * Config::Bfield);
      }
    }

    float kinv[NN];
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      kinv[n] = 1.f / k[n];
    }

#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      dprint_np(n,
                std::endl
                    << "input parameters"
                    << " inPar.constAt(n, 0, 0)=" << std::setprecision(9) << inPar.constAt(n, 0, 0)
                    << " inPar.constAt(n, 1, 0)=" << std::setprecision(9) << inPar.constAt(n, 1, 0)
                    << " inPar.constAt(n, 2, 0)=" << std::setprecision(9) << inPar.constAt(n, 2, 0)
                    << " inPar.constAt(n, 3, 0)=" << std::setprecision(9) << inPar.constAt(n, 3, 0)
                    << " inPar.constAt(n, 4, 0)=" << std::setprecision(9) << inPar.constAt(n, 4, 0)
                    << " inPar.constAt(n, 5, 0)=" << std::setprecision(9) << inPar.constAt(n, 5, 0)
                    << " inChg.constAt(n, 0, 0)=" << std::setprecision(9) << inChg.constAt(n, 0, 0));
    }
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      dprint_np(n,
                "propagation start, dump parameters"
                    << std::endl
                    << "pos = " << inPar.constAt(n, 0, 0) << " " << inPar.constAt(n, 1, 0) << " "
                    << inPar.constAt(n, 2, 0) << std::endl
                    << "mom (cart) = " << std::cos(inPar.constAt(n, 4, 0)) / inPar.constAt(n, 3, 0) << " "
                    << std::sin(inPar.constAt(n, 4, 0)) / inPar.constAt(n, 3, 0) << " "
                    << 1. / (inPar.constAt(n, 3, 0) * tan(inPar.constAt(n, 5, 0))) << " r="
                    << std::sqrt(inPar.constAt(n, 0, 0) * inPar.constAt(n, 0, 0) +
                                 inPar.constAt(n, 1, 0) * inPar.constAt(n, 1, 0))
                    << " pT=" << 1. / std::abs(inPar.constAt(n, 3, 0)) << " q=" << inChg.constAt(n, 0, 0)
                    << " targetZ=" << msZ.constAt(n, 0, 0) << std::endl);
    }

    float pt[NN];
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      pt[n] = 1.f / ipt[n];
    }

    //no trig approx here, phi can be large
    float cosP[NN];
    float sinP[NN];
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      cosP[n] = std::cos(phiin[n]);
    }

#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      sinP[n] = std::sin(phiin[n]);
    }

    float cosT[NN];
    float sinT[NN];
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      cosT[n] = std::cos(theta[n]);
    }

#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      sinT[n] = std::sin(theta[n]);
    }

    float tanT[NN];
    float icos2T[NN];
    float pxin[NN];
    float pyin[NN];
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      tanT[n] = sinT[n] / cosT[n];
      icos2T[n] = 1.f / (cosT[n] * cosT[n]);
      pxin[n] = cosP[n] * pt[n];
      pyin[n] = sinP[n] * pt[n];
    }

    float deltaZ[NN];
    float alpha[NN];
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      deltaZ[n] = zout[n] - zin[n];
      alpha[n] = deltaZ[n] * tanT[n] * ipt[n] * kinv[n];
    }

    float cosahTmp[NN];
    float sinahTmp[NN];
    if constexpr (Config::useTrigApprox) {
#if !defined(__INTEL_COMPILER)
#pragma omp simd
#endif
      for (int n = 0; n < NN; ++n) {
        sincos4(alpha[n] * 0.5f, sinahTmp[n], cosahTmp[n]);
      }
    } else {
#if !defined(__INTEL_COMPILER)
#pragma omp simd
#endif
      for (int n = 0; n < NN; ++n) {
        cosahTmp[n] = std::cos(alpha[n] * 0.5f);
      }
#if !defined(__INTEL_COMPILER)
#pragma omp simd
#endif
      for (int n = 0; n < NN; ++n) {
        sinahTmp[n] = std::sin(alpha[n] * 0.5f);
      }
    }

    float cosah[NN];
    float sinah[NN];
    float cosa[NN];
    float sina[NN];
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      cosah[n] = cosahTmp[n];
      sinah[n] = sinahTmp[n];
      cosa[n] = 1.f - 2.f * sinah[n] * sinah[n];
      sina[n] = 2.f * sinah[n] * cosah[n];
    }

//update parameters
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      outPar.At(n, 0, 0) = outPar.At(n, 0, 0) + 2.f * k[n] * sinah[n] * (pxin[n] * cosah[n] - pyin[n] * sinah[n]);
      outPar.At(n, 1, 0) = outPar.At(n, 1, 0) + 2.f * k[n] * sinah[n] * (pyin[n] * cosah[n] + pxin[n] * sinah[n]);
      outPar.At(n, 2, 0) = zout[n];
      outPar.At(n, 4, 0) = phiin[n] + alpha[n];
    }

#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      dprint_np(n,
                "propagation to Z end (OLD), dump parameters\n"
                    << "   pos = " << outPar(n, 0, 0) << " " << outPar(n, 1, 0) << " " << outPar(n, 2, 0) << "\t\t r="
                    << std::sqrt(outPar(n, 0, 0) * outPar(n, 0, 0) + outPar(n, 1, 0) * outPar(n, 1, 0)) << std::endl
                    << "   mom = " << outPar(n, 3, 0) << " " << outPar(n, 4, 0) << " " << outPar(n, 5, 0) << std::endl
                    << " cart= " << std::cos(outPar(n, 4, 0)) / outPar(n, 3, 0) << " "
                    << std::sin(outPar(n, 4, 0)) / outPar(n, 3, 0) << " "
                    << 1. / (outPar(n, 3, 0) * tan(outPar(n, 5, 0))) << "\t\tpT=" << 1. / std::abs(outPar(n, 3, 0))
                    << std::endl);
    }

    float pxcaMpysa[NN];
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      pxcaMpysa[n] = pxin[n] * cosa[n] - pyin[n] * sina[n];
    }

#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      errorProp(n, 0, 2) = -tanT[n] * ipt[n] * pxcaMpysa[n];
      errorProp(n, 0, 3) =
          k[n] * pt[n] * pt[n] *
          (cosP[n] * (alpha[n] * cosa[n] - sina[n]) + sinP[n] * 2.f * sinah[n] * (sinah[n] - alpha[n] * cosah[n]));
      errorProp(n, 0, 4) = -2.f * k[n] * pt[n] * sinah[n] * (sinP[n] * cosah[n] + cosP[n] * sinah[n]);
      errorProp(n, 0, 5) = deltaZ[n] * ipt[n] * pxcaMpysa[n] * icos2T[n];
    }

    float pycaPpxsa[NN];
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      pycaPpxsa[n] = pyin[n] * cosa[n] + pxin[n] * sina[n];
    }

#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      errorProp(n, 1, 2) = -tanT[n] * ipt[n] * pycaPpxsa[n];
      errorProp(n, 1, 3) =
          k[n] * pt[n] * pt[n] *
          (sinP[n] * (alpha[n] * cosa[n] - sina[n]) - cosP[n] * 2.f * sinah[n] * (sinah[n] - alpha[n] * cosah[n]));
      errorProp(n, 1, 4) = 2.f * k[n] * pt[n] * sinah[n] * (cosP[n] * cosah[n] - sinP[n] * sinah[n]);
      errorProp(n, 1, 5) = deltaZ[n] * ipt[n] * pycaPpxsa[n] * icos2T[n];
    }

#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      errorProp(n, 4, 2) = -ipt[n] * tanT[n] * kinv[n];
      errorProp(n, 4, 3) = tanT[n] * deltaZ[n] * kinv[n];
      errorProp(n, 4, 5) = ipt[n] * deltaZ[n] * kinv[n] * icos2T[n];
    }

#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      dprint_np(
          n,
          "propagation end, dump parameters"
              << std::endl
              << "pos = " << outPar.At(n, 0, 0) << " " << outPar.At(n, 1, 0) << " " << outPar.At(n, 2, 0) << std::endl
              << "mom (cart) = " << std::cos(outPar.At(n, 4, 0)) / outPar.At(n, 3, 0) << " "
              << std::sin(outPar.At(n, 4, 0)) / outPar.At(n, 3, 0) << " "
              << 1. / (outPar.At(n, 3, 0) * tan(outPar.At(n, 5, 0)))
              << " r=" << std::sqrt(outPar.At(n, 0, 0) * outPar.At(n, 0, 0) + outPar.At(n, 1, 0) * outPar.At(n, 1, 0))
              << " pT=" << 1. / std::abs(outPar.At(n, 3, 0)) << std::endl);
    }

    // PROP-FAIL-ENABLE Disabled to keep physics changes minimal.
    // To be reviewed, enabled and processed accordingly elsewhere.
    /*
    // Check for errors, set fail-flag.
    for (int n = 0; n < NN; ++n) {
      // We propagate for alpha: mark fail when prop angle more than pi/2
      if (std::abs(alpha[n]) > 1.57) {
        dprintf("helixAtZ: more than quarter turn, alpha = %f\n", alpha[n]);
        outFailFlag[n] = 1;
      } else {
        // Have we reached desired z? We can't know, we copy desired z to actual z.
        // Are we close to apex? Same condition as in propToR, 12.5 deg, cos(78.5deg) = 0.2
        float dotp = (outPar.At(n, 0, 0) * std::cos(outPar.At(n, 4, 0)) +
                      outPar.At(n, 1, 0) * std::sin(outPar.At(n, 4, 0))) /
                     hipo(outPar.At(n, 0, 0), outPar.At(n, 1, 0));
        if (dotp < 0.2 || dotp < 0) {
          dprintf("helixAtZ: dot product bad, dotp = %f\n", dotp);
          outFailFlag[n] = 1;
        }
      }
    }
    */

#ifdef DEBUG
    if (debug && g_debug) {
      for (int n = 0; n < N_proc; ++n) {
        dmutex_guard;
        std::cout << n << ": jacobian" << std::endl;
        printf("%5f %5f %5f %5f %5f %5f\n",
               errorProp(n, 0, 0),
               errorProp(n, 0, 1),
               errorProp(n, 0, 2),
               errorProp(n, 0, 3),
               errorProp(n, 0, 4),
               errorProp(n, 0, 5));
        printf("%5f %5f %5f %5f %5f %5f\n",
               errorProp(n, 1, 0),
               errorProp(n, 1, 1),
               errorProp(n, 1, 2),
               errorProp(n, 1, 3),
               errorProp(n, 1, 4),
               errorProp(n, 1, 5));
        printf("%5f %5f %5f %5f %5f %5f\n",
               errorProp(n, 2, 0),
               errorProp(n, 2, 1),
               errorProp(n, 2, 2),
               errorProp(n, 2, 3),
               errorProp(n, 2, 4),
               errorProp(n, 2, 5));
        printf("%5f %5f %5f %5f %5f %5f\n",
               errorProp(n, 3, 0),
               errorProp(n, 3, 1),
               errorProp(n, 3, 2),
               errorProp(n, 3, 3),
               errorProp(n, 3, 4),
               errorProp(n, 3, 5));
        printf("%5f %5f %5f %5f %5f %5f\n",
               errorProp(n, 4, 0),
               errorProp(n, 4, 1),
               errorProp(n, 4, 2),
               errorProp(n, 4, 3),
               errorProp(n, 4, 4),
               errorProp(n, 4, 5));
        printf("%5f %5f %5f %5f %5f %5f\n",
               errorProp(n, 5, 0),
               errorProp(n, 5, 1),
               errorProp(n, 5, 2),
               errorProp(n, 5, 3),
               errorProp(n, 5, 4),
               errorProp(n, 5, 5));
      }
    }
#endif
  }

}  // namespace mkfit
