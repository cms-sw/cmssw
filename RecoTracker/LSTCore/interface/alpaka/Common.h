#ifndef RecoTracker_LSTCore_interface_alpaka_Common_h
#define RecoTracker_LSTCore_interface_alpaka_Common_h

#include <numbers>

#include "FWCore/Utilities/interface/HostDeviceConstant.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoTracker/LSTCore/interface/Common.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  using namespace ::lst;

  ALPAKA_FN_HOST ALPAKA_FN_INLINE void lstWarning(std::string_view warning) {
#ifdef LST_STANDALONE
    printf("%s\n", warning.data());
#else
    edm::LogWarning("LST") << warning;
#endif
  }

  // The constants below are usually used in functions like alpaka::math::min(),
  // expecting a reference (T const&) in the arguments. Hence,
  // HOST_DEVICE_CONSTANT needs to be used instead of constexpr.

  HOST_DEVICE_CONSTANT float kPi = std::numbers::pi_v<float>;
  // 15 MeV constant from the approximate Bethe-Bloch formula
  HOST_DEVICE_CONSTANT float kMulsInGeV = 0.015;
  HOST_DEVICE_CONSTANT float kMiniMulsPtScaleBarrel[6] = {0.0052, 0.0038, 0.0034, 0.0034, 0.0032, 0.0034};
  HOST_DEVICE_CONSTANT float kMiniMulsPtScaleEndcap[5] = {0.006, 0.006, 0.006, 0.006, 0.006};
  HOST_DEVICE_CONSTANT float kMiniRminMeanBarrel[6] = {
      25.007152356, 37.2186993757, 52.3104270826, 68.6658656666, 85.9770373007, 108.301772384};
  HOST_DEVICE_CONSTANT float kMiniRminMeanEndcap[5] = {
      130.992832231, 154.813883559, 185.352604327, 221.635123002, 265.022076742};
  HOST_DEVICE_CONSTANT float k2Rinv1GeVf = (2.99792458e-3 * 3.8) / 2;
  HOST_DEVICE_CONSTANT float kR1GeVf = 1. / (2.99792458e-3 * 3.8);
  HOST_DEVICE_CONSTANT float kSinAlphaMax = 0.95;
  HOST_DEVICE_CONSTANT float kDeltaZLum = 15.0;
  HOST_DEVICE_CONSTANT float kPixelPSZpitch = 0.15;
  HOST_DEVICE_CONSTANT float kStripPSZpitch = 2.4;
  HOST_DEVICE_CONSTANT float kStrip2SZpitch = 5.0;
  HOST_DEVICE_CONSTANT float kWidth2S = 0.009;
  HOST_DEVICE_CONSTANT float kWidthPS = 0.01;
  HOST_DEVICE_CONSTANT float kPt_betaMax = 7.0;
  // To be updated with std::numeric_limits<float>::infinity() in the code and data files
  HOST_DEVICE_CONSTANT float kVerticalModuleSlope = 123456789.0;

  HOST_DEVICE_CONSTANT float kMiniDeltaTilted[3] = {0.26f, 0.26f, 0.26f};
  HOST_DEVICE_CONSTANT float kMiniDeltaFlat[6] = {0.26f, 0.16f, 0.16f, 0.18f, 0.18f, 0.18f};
  HOST_DEVICE_CONSTANT float kMiniDeltaLooseTilted[3] = {0.4f, 0.4f, 0.4f};
  HOST_DEVICE_CONSTANT float kMiniDeltaEndcap[5][15] = {
      {0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, /*10*/ 0.18f, 0.18f, 0.18f, 0.18f, 0.18f},
      {0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, /*10*/ 0.18f, 0.18f, 0.18f, 0.18f, 0.18f},
      {0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.18f, 0.18f, /*10*/ 0.18f, 0.18f, 0.18f, 0.18f, 0.18f},
      {0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.18f, 0.18f, /*10*/ 0.18f, 0.18f, 0.18f, 0.18f, 0.18f},
      {0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.18f, /*10*/ 0.18f, 0.18f, 0.18f, 0.18f, 0.18f}};

  namespace dnn {

    // Common constants for both DNNs
    HOST_DEVICE_CONSTANT float kPhi_norm = kPi;
    HOST_DEVICE_CONSTANT float kEtaSize = 0.25f;  // Bin size in eta.
    constexpr unsigned int kPtBins = 2;
    constexpr unsigned int kEtaBins = 10;

    namespace plsembdnn {
      HOST_DEVICE_CONSTANT float kEta_norm = 4.0f;
      HOST_DEVICE_CONSTANT float kEtaErr_norm = 0.00139f;
      HOST_DEVICE_CONSTANT float kWP[kEtaBins] = {
          0.9235f, 0.8974f, 0.9061f, 0.9431f, 0.8262f, 0.7998f, 0.7714f, 0.7017f, 0.6749f, 0.6624f};
    }  // namespace plsembdnn

    namespace t3dnn {
      HOST_DEVICE_CONSTANT float kEta_norm = 2.5f;
      HOST_DEVICE_CONSTANT float kZ_max = 224.149505f;
      HOST_DEVICE_CONSTANT float kR_max = 98.932365f;
      HOST_DEVICE_CONSTANT unsigned int kOutputFeatures = 3;
      HOST_DEVICE_CONSTANT float kWp_prompt[kPtBins][kEtaBins] = {
          {0.4957f, 0.5052f, 0.5201f, 0.5340f, 0.4275f, 0.4708f, 0.4890f, 0.4932f, 0.5400f, 0.5449f},
          {0.0302f, 0.0415f, 0.0994f, 0.1791f, 0.1960f, 0.2467f, 0.3227f, 0.3242f, 0.2367f, 0.2187f}};
      HOST_DEVICE_CONSTANT float kWp_displaced[kPtBins][kEtaBins] = {
          {0.0334f, 0.0504f, 0.0748f, 0.0994f, 0.1128f, 0.1123f, 0.1118f, 0.1525f, 0.1867f, 0.1847f},
          {0.0091f, 0.0075f, 0.0350f, 0.0213f, 0.0435f, 0.0676f, 0.1957f, 0.1649f, 0.1080f, 0.1046f}};
    }  // namespace t3dnn

    namespace t5dnn {
      HOST_DEVICE_CONSTANT float kEta_norm = 2.5f;
      HOST_DEVICE_CONSTANT float kZ_max = 267.2349854f;
      HOST_DEVICE_CONSTANT float kR_max = 110.1099396f;
      HOST_DEVICE_CONSTANT float kWp[kPtBins][kEtaBins] = {
          {0.4493f, 0.4939f, 0.5715f, 0.6488f, 0.5709f, 0.5938f, 0.7164f, 0.7565f, 0.8103f, 0.8593f},
          {0.4488f, 0.4448f, 0.5067f, 0.5929f, 0.4836f, 0.4112f, 0.4968f, 0.4403f, 0.5597f, 0.5067f}};
    }  // namespace t5dnn

    namespace pt3dnn {
      HOST_DEVICE_CONSTANT float kEta_norm = 2.5f;
      HOST_DEVICE_CONSTANT float kWp[kEtaBins] = {
          0.189f, 0.1805f, 0.2267f, 0.3104f, 0.4719f, 0.3159f, 0.1372f, 0.1571f, 0.3198f, 0.186f};
      HOST_DEVICE_CONSTANT float kWpHigh = 0.0473f;
    }  // namespace pt3dnn

  }  // namespace dnn

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
