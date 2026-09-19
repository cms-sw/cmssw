#ifndef RecoTracker_FinalTrackSelectors_plugins_alpaka_PixelTrackForestHighPuritySelectorKernels_h
#define RecoTracker_FinalTrackSelectors_plugins_alpaka_PixelTrackForestHighPuritySelectorKernels_h

#include <alpaka/alpaka.hpp>
#include <array>
#include <type_traits>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "RecoTracker/FinalTrackSelectors/interface/PixelTrackFeaturesSoA.h"

// The forest reads the full PixelTrackFeaturesSoA width.
inline constexpr int kNForestFeatures = kNPixelTrackFeatures;

// One named float per SoA column; asArray() gives them in column order, which the model's split
// feature index addresses.
struct PixelTrackForestFeatures {
  float chi2 = 0.f;
  float dzError = 0.f;
  float dxyError = 0.f;
  float eta = 0.f;
  float nHits = 0.f;
  float phi = 0.f;
  float phiError = 0.f;
  float pt = 0.f;
  float qOverPtError = 0.f;
  float dzBS = 0.f;
  float dxyBS = 0.f;
  float nLayers = 0.f;
  float cotThetaError = 0.f;
  float covCotThetaDz = 0.f;
  float covDxyQOverPt = 0.f;
  float covPhiDxy = 0.f;
  float covPhiQOverPt = 0.f;
  float caFitChi2 = 0.f;
  float psFrac = 0.f;
  float r0 = 0.f;
  float nPS = 0.f;
  float spanZ = 0.f;
  float nStubs = 0.f;
  float logChi2Stub = 0.f;
  float kErr = 0.f;
  float dcaEst = 0.f;
  float nBarrel = 0.f;
  float rzChi2 = 0.f;
  float meanStubKappa = 0.f;
  float leverArm = 0.f;
  float rMax = 0.f;
  float nAttached = 0.f;
  float nOTExtras = 0.f;
  float iterationId = 0.f;
  float ndof = 0.f;
  float minCharge = 0.f;
  float meanCharge = 0.f;
  float minChargeNorm = 0.f;
  float maxSizeY = 0.f;
  float meanSizeY = 0.f;
  float maxSizeX = 0.f;
  float nLowCharge = 0.f;

  ALPAKA_FN_HOST_ACC constexpr std::array<float, kNForestFeatures> asArray() const {
    return {chi2,          dzError,     dxyError, eta,       nHits,         phi,           phiError,      pt,
            qOverPtError,  dzBS,        dxyBS,    nLayers,   cotThetaError, covCotThetaDz, covDxyQOverPt, covPhiDxy,
            covPhiQOverPt, caFitChi2,   psFrac,   r0,        nPS,           spanZ,         nStubs,        logChi2Stub,
            kErr,          dcaEst,      nBarrel,  rzChi2,    meanStubKappa, leverArm,      rMax,          nAttached,
            nOTExtras,     iterationId, ndof,     minCharge, meanCharge,    minChargeNorm, maxSizeY,      meanSizeY,
            maxSizeX,      nLowCharge};
  }
};
static_assert(std::is_standard_layout_v<PixelTrackForestFeatures>);

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  void launchTreeScore(Queue& queue,
                       const int maxPreselectedTracks,
                       const int8_t* treeFeat,
                       const float* treeVal,
                       const int32_t* treeLeft,
                       const int32_t* treeRight,
                       const int32_t* treeRoots,
                       const int nTrees,
                       const float baseLogit,
                       const PixelTrackFeaturesSoA::ConstView trackFeatures,
                       const int* nPreselectedTracks,
                       PixelTrackScoresSoA::View trackScores);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_FinalTrackSelectors_plugins_alpaka_PixelTrackForestHighPuritySelectorKernels_h
