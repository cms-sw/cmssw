#ifndef RecoTracker_LSTCore_interface_alpaka_Common_h
#define RecoTracker_LSTCore_interface_alpaka_Common_h

#include <numbers>

#include "RecoTracker/LSTCore/interface/Common.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  using namespace ::lst;

  Vec3D constexpr elementsPerThread(Vec3D::all(static_cast<Idx>(1)));

  // Adjust grid and block sizes based on backend configuration
  template <typename Vec, typename TAcc = Acc<typename Vec::Dim>>
  ALPAKA_FN_HOST ALPAKA_FN_INLINE WorkDiv<typename Vec::Dim> createWorkDiv(const Vec& blocksPerGrid,
                                                                           const Vec& threadsPerBlock,
                                                                           const Vec& elementsPerThreadArg) {
    Vec adjustedBlocks = blocksPerGrid;
    Vec adjustedThreads = threadsPerBlock;

    // special overrides for CPU/host cases
    if constexpr (std::is_same_v<Platform, alpaka::PlatformCpu>) {
      adjustedBlocks = Vec::all(static_cast<Idx>(1));

      if constexpr (alpaka::accMatchesTags<TAcc, alpaka::TagCpuSerial>) {
        // Serial execution, set threads to 1 as well
        adjustedThreads = Vec::all(static_cast<Idx>(1));  // probably redundant
      }
    }

    return WorkDiv<typename Vec::Dim>(adjustedBlocks, adjustedThreads, elementsPerThreadArg);
  }

  // The constants below are usually used in functions like alpaka::math::min(),
  // expecting a reference (T const&) in the arguments. Hence,
  // ALPAKA_STATIC_ACC_MEM_GLOBAL needs to be used in addition to constexpr.

  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kPi = std::numbers::pi_v<float>;
  // 15 MeV constant from the approximate Bethe-Bloch formula
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kMulsInGeV = 0.015;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kMiniMulsPtScaleBarrel[6] = {
      0.0052, 0.0038, 0.0034, 0.0034, 0.0032, 0.0034};
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kMiniMulsPtScaleEndcap[5] = {0.006, 0.006, 0.006, 0.006, 0.006};
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kMiniRminMeanBarrel[6] = {
      25.007152356, 37.2186993757, 52.3104270826, 68.6658656666, 85.9770373007, 108.301772384};
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kMiniRminMeanEndcap[5] = {
      130.992832231, 154.813883559, 185.352604327, 221.635123002, 265.022076742};
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float k2Rinv1GeVf = (2.99792458e-3 * 3.8) / 2;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kR1GeVf = 1. / (2.99792458e-3 * 3.8);
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kSinAlphaMax = 0.95;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float ptCut = PT_CUT;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kDeltaZLum = 15.0;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kPixelPSZpitch = 0.15;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kStripPSZpitch = 2.4;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kStrip2SZpitch = 5.0;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWidth2S = 0.009;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWidthPS = 0.01;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kPt_betaMax = 7.0;
  // To be updated with std::numeric_limits<float>::infinity() in the code and data files
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kVerticalModuleSlope = 123456789.0;

  namespace t5dnn {

    // Working points matching LST fake rate (43.9%) or signal acceptance (82.0%)
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kLSTWp1 = 0.3418833f;  // 94.0% TPR, 43.9% FPR
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kLSTWp2 = 0.6177366f;  // 82.0% TPR, 20.0% FPR
    // Other working points
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWp70 = 0.7776195f;    // 70.0% TPR, 10.0% FPR
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWp75 = 0.7181118f;    // 75.0% TPR, 13.5% FPR
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWp80 = 0.6492643f;    // 80.0% TPR, 17.9% FPR
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWp85 = 0.5655319f;    // 85.0% TPR, 23.8% FPR
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWp90 = 0.4592205f;    // 90.0% TPR, 32.6% FPR
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWp95 = 0.3073708f;    // 95.0% TPR, 47.7% FPR
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWp97p5 = 0.2001348f;  // 97.5% TPR, 61.2% FPR
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWp99 = 0.1120605f;    // 99.0% TPR, 75.9% FPR
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWp99p9 = 0.0218196f;  // 99.9% TPR, 95.4% FPR

  }  // namespace t5dnn

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
