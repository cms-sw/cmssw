#ifndef RecoTracker_LSTCore_interface_alpaka_Common_h
#define RecoTracker_LSTCore_interface_alpaka_Common_h

#include <numbers>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoTracker/LSTCore/interface/Common.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  using namespace ::lst;

  Vec3D constexpr elementsPerThread(Vec3D::all(static_cast<Idx>(1)));

  ALPAKA_FN_HOST ALPAKA_FN_INLINE void lstWarning(std::string warning) {
    edm::LogWarning("LST") << warning;
    return;
  }

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
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kDeltaZLum = 15.0;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kPixelPSZpitch = 0.15;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kStripPSZpitch = 2.4;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kStrip2SZpitch = 5.0;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWidth2S = 0.009;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWidthPS = 0.01;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kPt_betaMax = 7.0;
  // To be updated with std::numeric_limits<float>::infinity() in the code and data files
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kVerticalModuleSlope = 123456789.0;

  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kMiniDeltaTilted[3] = {0.26f, 0.26f, 0.26f};
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kMiniDeltaFlat[6] = {0.26f, 0.16f, 0.16f, 0.18f, 0.18f, 0.18f};
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kMiniDeltaLooseTilted[3] = {0.4f, 0.4f, 0.4f};
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kMiniDeltaEndcap[5][15] = {
      {0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, /*10*/ 0.18f, 0.18f, 0.18f, 0.18f, 0.18f},
      {0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, /*10*/ 0.18f, 0.18f, 0.18f, 0.18f, 0.18f},
      {0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.18f, 0.18f, /*10*/ 0.18f, 0.18f, 0.18f, 0.18f, 0.18f},
      {0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.18f, 0.18f, /*10*/ 0.18f, 0.18f, 0.18f, 0.18f, 0.18f},
      {0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.18f, /*10*/ 0.18f, 0.18f, 0.18f, 0.18f, 0.18f}};

  namespace t5dnn {
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kZ_max = 267.2349854f;
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kR_max = 110.1099396f;
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kEta_norm = 2.5f;
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kPhi_norm = kPi;
    // pt, eta binned
    constexpr unsigned int kPtBins = 2;
    constexpr unsigned int kEtaBins = 10;
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWp[kPtBins][kEtaBins] = {
        {0.4493, 0.4939, 0.5715, 0.6488, 0.5709, 0.5938, 0.7164, 0.7565, 0.8103, 0.8593},
        {0.4488, 0.4448, 0.5067, 0.5929, 0.4836, 0.4112, 0.4968, 0.4403, 0.5597, 0.5067}};
  }  // namespace t5dnn

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
