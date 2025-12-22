#ifndef RecoTracker_LSTCore_src_alpaka_Quadruplet_h
#define RecoTracker_LSTCore_src_alpaka_Quadruplet_h

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "FWCore/Utilities/interface/CMSUnrollLoop.h"

#include "RecoTracker/LSTCore/interface/ObjectRangesSoA.h"
#include "RecoTracker/LSTCore/interface/MiniDoubletsSoA.h"
#include "RecoTracker/LSTCore/interface/SegmentsSoA.h"
#include "RecoTracker/LSTCore/interface/TripletsSoA.h"
#include "RecoTracker/LSTCore/interface/QuadrupletsSoA.h"
#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/ModulesSoA.h"
#include "RecoTracker/LSTCore/interface/EndcapGeometry.h"
#include "RecoTracker/LSTCore/interface/ObjectRangesSoA.h"
#include "RecoTracker/LSTCore/interface/Circle.h"

#include "Quintuplet.h"

#include "NeuralNetwork.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addQuadrupletToMemory(TripletsConst triplets,
                                                            Quadruplets quadruplets,
                                                            unsigned int innerTripletIndex,
                                                            unsigned int outerTripletIndex,
                                                            uint16_t lowerModule1,
                                                            uint16_t lowerModule2,
                                                            uint16_t lowerModule3,
                                                            uint16_t lowerModule4,
                                                            float innerRadius,
                                                            float outerRadius,
                                                            float pt,
                                                            float eta,
                                                            float phi,
                                                            float scores,
                                                            uint8_t layer,
                                                            unsigned int quadrupletIndex,
                                                            float rzChiSquared,
                                                            float dBeta,
                                                            float promptScore,
                                                            float displacedScore,
                                                            float fakeScore,
                                                            float regressionCenterX,
                                                            float regressionCenterY,
                                                            float regressionRadius,
                                                            float nonAnchorRegressionRadius) {
    quadruplets.tripletIndices()[quadrupletIndex][0] = innerTripletIndex;
    quadruplets.tripletIndices()[quadrupletIndex][1] = outerTripletIndex;

    quadruplets.lowerModuleIndices()[quadrupletIndex][0] = lowerModule1;
    quadruplets.lowerModuleIndices()[quadrupletIndex][1] = lowerModule2;
    quadruplets.lowerModuleIndices()[quadrupletIndex][2] = lowerModule3;
    quadruplets.lowerModuleIndices()[quadrupletIndex][3] = lowerModule4;
    quadruplets.innerRadius()[quadrupletIndex] = __F2H(innerRadius);
    quadruplets.outerRadius()[quadrupletIndex] = __F2H(outerRadius);
    quadruplets.pt()[quadrupletIndex] = __F2H(pt);
    quadruplets.eta()[quadrupletIndex] = __F2H(eta);
    quadruplets.phi()[quadrupletIndex] = __F2H(phi);
    quadruplets.score_rphisum()[quadrupletIndex] = __F2H(scores);
    quadruplets.layer()[quadrupletIndex] = layer;
    quadruplets.isDup()[quadrupletIndex] = 0;
    quadruplets.logicalLayers()[quadrupletIndex][0] = triplets.logicalLayers()[innerTripletIndex][0];
    quadruplets.logicalLayers()[quadrupletIndex][1] = triplets.logicalLayers()[innerTripletIndex][1];
    quadruplets.logicalLayers()[quadrupletIndex][2] = triplets.logicalLayers()[innerTripletIndex][2];
    quadruplets.logicalLayers()[quadrupletIndex][3] = triplets.logicalLayers()[outerTripletIndex][2];

    quadruplets.hitIndices()[quadrupletIndex][0] = triplets.hitIndices()[innerTripletIndex][0];
    quadruplets.hitIndices()[quadrupletIndex][1] = triplets.hitIndices()[innerTripletIndex][1];
    quadruplets.hitIndices()[quadrupletIndex][2] = triplets.hitIndices()[innerTripletIndex][2];
    quadruplets.hitIndices()[quadrupletIndex][3] = triplets.hitIndices()[innerTripletIndex][3];
    quadruplets.hitIndices()[quadrupletIndex][4] = triplets.hitIndices()[innerTripletIndex][4];
    quadruplets.hitIndices()[quadrupletIndex][5] = triplets.hitIndices()[innerTripletIndex][5];
    quadruplets.hitIndices()[quadrupletIndex][6] = triplets.hitIndices()[outerTripletIndex][4];
    quadruplets.hitIndices()[quadrupletIndex][7] = triplets.hitIndices()[outerTripletIndex][5];

    quadruplets.rzChiSquared()[quadrupletIndex] = rzChiSquared;
    quadruplets.dBeta()[quadrupletIndex] = dBeta;
    quadruplets.promptScore()[quadrupletIndex] = promptScore;
    quadruplets.displacedScore()[quadrupletIndex] = displacedScore;
    quadruplets.fakeScore()[quadrupletIndex] = fakeScore;

    quadruplets.regressionRadius()[quadrupletIndex] = regressionRadius;
    quadruplets.nonAnchorRegressionRadius()[quadrupletIndex] = nonAnchorRegressionRadius;
    quadruplets.regressionCenterX()[quadrupletIndex] = regressionCenterX;
    quadruplets.regressionCenterY()[quadrupletIndex] = regressionCenterY;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passT4RZConstraint(TAcc const& acc,
                                                         ModulesConst modules,
                                                         MiniDoubletsConst mds,
                                                         unsigned int firstMDIndex,
                                                         unsigned int secondMDIndex,
                                                         unsigned int thirdMDIndex,
                                                         unsigned int fourthMDIndex,
                                                         uint16_t lowerModuleIndex1,
                                                         uint16_t lowerModuleIndex2,
                                                         uint16_t lowerModuleIndex3,
                                                         uint16_t lowerModuleIndex4,
                                                         float& rzChiSquared,
                                                         float inner_pt,
                                                         float innerRadius,
                                                         float g,
                                                         float f,
                                                         short charge) {
    //all the values are stored in the unit of cm, in the calculation below we need to be cautious if we want to use the meter unit
    //get r and z
    const float rt1 = mds.anchorRt()[firstMDIndex] / 100;
    const float rt2 = mds.anchorRt()[secondMDIndex] / 100;
    const float rt3 = mds.anchorRt()[thirdMDIndex] / 100;
    const float rt4 = mds.anchorRt()[fourthMDIndex] / 100;

    const float z1 = mds.anchorZ()[firstMDIndex] / 100;
    const float z2 = mds.anchorZ()[secondMDIndex] / 100;
    const float z3 = mds.anchorZ()[thirdMDIndex] / 100;
    const float z4 = mds.anchorZ()[fourthMDIndex] / 100;

    // Using lst_layer numbering convention defined in ModuleMethods.h
    const short layer2 = modules.lstLayers()[lowerModuleIndex2];
    const short layer3 = modules.lstLayers()[lowerModuleIndex3];
    const short layer4 = modules.lstLayers()[lowerModuleIndex4];

    // Get the module type of each MD: 0 is ps, 1 is 2s
    const bool moduleType1 = modules.moduleType()[lowerModuleIndex1];
    const bool moduleType2 = modules.moduleType()[lowerModuleIndex2];
    const bool moduleType3 = modules.moduleType()[lowerModuleIndex3];
    const bool moduleType4 = modules.moduleType()[lowerModuleIndex4];

    // Get the x,y position of each MD
    const float x1 = mds.anchorX()[firstMDIndex] / 100;
    const float x2 = mds.anchorX()[secondMDIndex] / 100;
    const float x3 = mds.anchorX()[thirdMDIndex] / 100;
    const float x4 = mds.anchorX()[fourthMDIndex] / 100;
    const float y1 = mds.anchorY()[firstMDIndex] / 100;
    const float y2 = mds.anchorY()[secondMDIndex] / 100;
    const float y3 = mds.anchorY()[thirdMDIndex] / 100;
    const float y4 = mds.anchorY()[fourthMDIndex] / 100;

    float residual = 0;
    float error2 = 0;
    // (g,f) is the center of the circle fitted by the innermost 3 points on x,y coordinates
    float x_center = g / 100, y_center = f / 100;
    float x_init = x3;
    float y_init = y3;
    float z_init = z3;
    float rt_init = rt3;  //use the third MD as initial point

    if (moduleType3 == 1)  // if MD3 is in 2s layer, use MD2 as initial point
    {
      x_init = x2;
      y_init = y2;
      z_init = z2;
      rt_init = rt2;
    }

    float pt = inner_pt, px = pt * charge * (y_init - y_center) / innerRadius * 100,
          py = -pt * charge * (x_init - x_center) / innerRadius * 100;

    // But if the initial T4 curve goes across quarters(i.e. cross axis to separate the quarters), need special redeclaration of px,py signs on these to avoid errors
    if (moduleType3 == 0) {  // 0 is ps
      if (x4 < x3 && x3 < x2)
        px = -alpaka::math::abs(acc, px);
      else if (x4 > x3 && x3 > x2)
        px = alpaka::math::abs(acc, px);
      if (y4 < y3 && y3 < y2)
        py = -alpaka::math::abs(acc, py);
      else if (y4 > y3 && y3 > y2)
        py = alpaka::math::abs(acc, py);
    } else if (moduleType3 == 1)  // 1 is 2s
    {
      if (x3 < x2 && x2 < x1)
        px = -alpaka::math::abs(acc, px);
      else if (x3 > x2 && x2 > x1)
        px = alpaka::math::abs(acc, px);
      if (y3 < y2 && y2 < y1)
        py = -alpaka::math::abs(acc, py);
      else if (y3 > y2 && y2 > y1)
        py = alpaka::math::abs(acc, py);
    }

    //to get pz, we use pt/pz=ds/dz, ds is the arclength between MD1 and MD3.
    float AO = alpaka::math::sqrt(acc, (x1 - x_center) * (x1 - x_center) + (y1 - y_center) * (y1 - y_center));
    float BO =
        alpaka::math::sqrt(acc, (x_init - x_center) * (x_init - x_center) + (y_init - y_center) * (y_init - y_center));
    float AB2 = (x1 - x_init) * (x1 - x_init) + (y1 - y_init) * (y1 - y_init);
    float dPhi = alpaka::math::acos(acc, (AO * AO + BO * BO - AB2) / (2 * AO * BO));
    float ds = innerRadius / 100 * dPhi;

    float pz = (z_init - z1) / ds * pt;
    float p = alpaka::math::sqrt(acc, px * px + py * py + pz * pz);

    float a = -2.f * k2Rinv1GeVf * 100 * charge;

    float zsi, rtsi;
    short layeri;
    bool moduleTypei;
    rzChiSquared = 0;
    float zs[] = {z2, z3, z4}, rts[] = {rt2, rt3, rt4};
    short layers[] = {layer2, layer3, layer4};
    bool moduleTypes[] = {moduleType2, moduleType3, moduleType4};
    for (size_t i = 2; i < 5; i++) {
      size_t j = i - 2;
      zsi = zs[j];
      rtsi = rts[j];
      layeri = layers[j];
      moduleTypei = moduleTypes[j];

      if (moduleType3 == 0) {  //0: ps
        if (i == 3)
          continue;
      } else {
        if (i == 2)
          continue;
      }
      // calculation is copied from PixelTriplet.cc computePT3RZChiSquared
      float diffr = 0, diffz = 0;

      float rou = a / p;
      // for endcap
      float s = (zsi - z_init) * p / pz;
      float x = x_init + px / a * alpaka::math::sin(acc, rou * s) - py / a * (1 - alpaka::math::cos(acc, rou * s));
      float y = y_init + py / a * alpaka::math::sin(acc, rou * s) + px / a * (1 - alpaka::math::cos(acc, rou * s));
      diffr = (rtsi - alpaka::math::sqrt(acc, x * x + y * y)) * 100;

      // for barrel
      if (layeri <= 6) {
        float paraA =
            rt_init * rt_init + 2 * (px * px + py * py) / (a * a) + 2 * (y_init * px - x_init * py) / a - rtsi * rtsi;
        float paraB = 2 * (x_init * px + y_init * py) / a;
        float paraC = 2 * (y_init * px - x_init * py) / a + 2 * (px * px + py * py) / (a * a);
        float A = paraB * paraB + paraC * paraC;
        float B = 2 * paraA * paraB;
        float C = paraA * paraA - paraC * paraC;
        float sol1 = (-B + alpaka::math::sqrt(acc, B * B - 4 * A * C)) / (2 * A);
        float sol2 = (-B - alpaka::math::sqrt(acc, B * B - 4 * A * C)) / (2 * A);
        float solz1 = alpaka::math::asin(acc, sol1) / rou * pz / p + z_init;
        float solz2 = alpaka::math::asin(acc, sol2) / rou * pz / p + z_init;
        float diffz1 = (solz1 - zsi) * 100;
        float diffz2 = (solz2 - zsi) * 100;
        diffz = edm::isNotFinite(diffz1) ? diffz2
                : edm::isNotFinite(diffz2)
                    ? diffz1
                    : ((alpaka::math::abs(acc, diffz1) < alpaka::math::abs(acc, diffz2)) ? diffz1 : diffz2);
      }
      residual = (layeri > 6) ? diffr : diffz;

      // error
      error2 = moduleTypei == 0 ? kPixelPSZpitch * kPixelPSZpitch : kStrip2SZpitch * kStrip2SZpitch;

      //check the tilted module, side: PosZ, NegZ, Center(for not tilted)
      float drdz;
      short side, subdets;
      if (i == 2) {
        drdz = alpaka::math::abs(acc, modules.drdzs()[lowerModuleIndex2]);
        side = modules.sides()[lowerModuleIndex2];
        subdets = modules.subdets()[lowerModuleIndex2];
      }
      if (i == 3) {
        drdz = alpaka::math::abs(acc, modules.drdzs()[lowerModuleIndex3]);
        side = modules.sides()[lowerModuleIndex3];
        subdets = modules.subdets()[lowerModuleIndex3];
      }
      const bool isEndcapOrCenter = (subdets == lst::Endcap) or (side == lst::Center);
      if (i == 2 || i == 3) {
        residual = (layeri <= 6 && ((side == Center) or (drdz < 1))) ? diffz : diffr;
        float projection_missing2 = 1.f;
        if (drdz < 1)
          projection_missing2 = isEndcapOrCenter ? 1.f : 1 / (1 + drdz * drdz);  // cos(atan(drdz)), if dr/dz<1
        if (drdz > 1)
          projection_missing2 =
              isEndcapOrCenter ? 1.f : (drdz * drdz) / (1 + drdz * drdz);  //sin(atan(drdz)), if dr/dz>1
        error2 = error2 * projection_missing2;
      }
      rzChiSquared += 12 * (residual * residual) / error2;
    }
    // for set rzchi2 cut
    // if the 4 points are linear, helix calculation gives nan
    if (inner_pt > 100 || edm::isNotFinite(rzChiSquared)) {
      float slope;
      const bool isPSPS2S = moduleType1 == 0 and moduleType2 == 0 and moduleType3 == 1;
      slope = isPSPS2S ? (z2 - z1) / (rt2 - rt1) : (z3 - z1) / (rt3 - rt1);
      float residual4_linear = (layer4 <= 6) ? ((z4 - z1) - slope * (rt4 - rt1)) : ((rt4 - rt1) - (z4 - z1) / slope);

      // creating a chi squared type quantity
      // 0-> PS, 1->2S
      residual4_linear = (moduleType4 == 0) ? residual4_linear / kPixelPSZpitch : residual4_linear / kStrip2SZpitch;
      residual4_linear = residual4_linear * 100;

      rzChiSquared = 12 * (residual4_linear * residual4_linear);
      return rzChiSquared < 5.839f;
    }
    float eta1 = alpaka::math::abs(acc, mds.anchorEta()[firstMDIndex]);
    uint8_t bin_index = (eta1 > 2.5f) ? (25 - 1) : static_cast<unsigned int>(eta1 / 0.1f);
    float chi2_cuts[] = {31.5082, 24.5654, 28.9223, 35.5906, 32.0746, 22.6416, 39.1476, 41.0791, 30.2745,
                         40.2882, 31.2135, 17.8911, 9.0297,  7.6862,  2.7591,  5.0587,  6.4014,  3.7348,
                         4.4768,  5.3087,  15.4535, 14.1107, 23.2778, 18.3643, 26.3276};
    return rzChiSquared < chi2_cuts[bin_index];
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuadrupletDefaultAlgo(TAcc const& acc,
                                                               ModulesConst modules,
                                                               MiniDoubletsConst mds,
                                                               SegmentsConst segments,
                                                               TripletsConst triplets,
                                                               uint16_t lowerModuleIndex1,
                                                               uint16_t lowerModuleIndex2,
                                                               uint16_t lowerModuleIndex3,
                                                               uint16_t lowerModuleIndex4,
                                                               unsigned int innerTripletIndex,
                                                               unsigned int outerTripletIndex,
                                                               float& regressionCenterX,
                                                               float& regressionCenterY,
                                                               float& regressionRadius,
                                                               float& nonAnchorRegressionRadius,
                                                               float& chiSquared,
                                                               const float ptCut,
                                                               float& rzChiSquared,
                                                               float& nonAnchorChiSquared,
                                                               float& dBeta,
                                                               float& promptScore,
                                                               float& displacedScore,
                                                               float& fakeScore) {
    unsigned int firstSegmentIndex = triplets.segmentIndices()[innerTripletIndex][0];
    unsigned int secondSegmentIndex = triplets.segmentIndices()[innerTripletIndex][1];
    unsigned int thirdSegmentIndex = triplets.segmentIndices()[outerTripletIndex][1];

    // require both T3s to have the same charge
    const short innerT3charge = triplets.charge()[innerTripletIndex];
    const short outerT3charge = triplets.charge()[outerTripletIndex];
    if (innerT3charge != outerT3charge)
      return false;

    unsigned int firstMDIndex = segments.mdIndices()[firstSegmentIndex][0];
    unsigned int secondMDIndex = segments.mdIndices()[secondSegmentIndex][0];
    unsigned int thirdMDIndex = segments.mdIndices()[secondSegmentIndex][1];
    unsigned int fourthMDIndex = segments.mdIndices()[thirdSegmentIndex][1];

    float x1 = mds.anchorX()[firstMDIndex];
    float x2 = mds.anchorX()[secondMDIndex];
    float x3 = mds.anchorX()[thirdMDIndex];
    float x4 = mds.anchorX()[fourthMDIndex];

    float y1 = mds.anchorY()[firstMDIndex];
    float y2 = mds.anchorY()[secondMDIndex];
    float y3 = mds.anchorY()[thirdMDIndex];
    float y4 = mds.anchorY()[fourthMDIndex];

    float inner_circleCenterX = triplets.centerX()[innerTripletIndex];
    float inner_circleCenterY = triplets.centerY()[innerTripletIndex];
    float innerRadius = triplets.radius()[innerTripletIndex];
    float outerRadius = triplets.radius()[outerTripletIndex];
    float inner_pt = 2 * k2Rinv1GeVf * innerRadius;
    float pt = (innerRadius + outerRadius) * k2Rinv1GeVf;

    // 4 categories for sigmas
    float sigmas2[4], delta1[4], delta2[4], slopes[4];
    bool isFlat[4];

    float xVec[] = {x1, x2, x3, x4};
    float yVec[] = {y1, y2, y3, y4};

    const uint16_t lowerModuleIndices[] = {lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4};

    computeSigmasForRegression(acc, modules, lowerModuleIndices, delta1, delta2, slopes, isFlat, Params_T4::kLayers);
    regressionRadius = computeRadiusUsingRegression(acc,
                                                    Params_T4::kLayers,
                                                    xVec,
                                                    yVec,
                                                    delta1,
                                                    delta2,
                                                    slopes,
                                                    isFlat,
                                                    regressionCenterX,
                                                    regressionCenterY,
                                                    sigmas2,
                                                    chiSquared);

    //compute the other chisquared
    //non anchor is always shifted for tilted and endcap!
    float nonAnchorSigmas2[4], nonAnchorDelta1[Params_T4::kLayers], nonAnchorDelta2[Params_T4::kLayers],
        nonAnchorSlopes[Params_T4::kLayers];
    float nonAnchorxs[] = {mds.outerX()[firstMDIndex],
                           mds.outerX()[secondMDIndex],
                           mds.outerX()[thirdMDIndex],
                           mds.outerX()[fourthMDIndex]};
    float nonAnchorys[] = {mds.outerY()[firstMDIndex],
                           mds.outerY()[secondMDIndex],
                           mds.outerY()[thirdMDIndex],
                           mds.outerY()[fourthMDIndex]};

    bool nonAnchorisFlat[4];
    float nonAnchorRegressionG, nonAnchorRegressionF;

    computeSigmasForRegression(acc,
                               modules,
                               lowerModuleIndices,
                               nonAnchorDelta1,
                               nonAnchorDelta2,
                               nonAnchorSlopes,
                               nonAnchorisFlat,
                               Params_T4::kLayers,
                               false);

    nonAnchorRegressionRadius = computeRadiusUsingRegression(acc,
                                                             Params_T4::kLayers,
                                                             nonAnchorxs,
                                                             nonAnchorys,
                                                             nonAnchorDelta1,
                                                             nonAnchorDelta2,
                                                             nonAnchorSlopes,
                                                             nonAnchorisFlat,
                                                             nonAnchorRegressionG,
                                                             nonAnchorRegressionF,
                                                             nonAnchorSigmas2,
                                                             nonAnchorChiSquared);

    bool inference = lst::t4dnn::runInference(acc,
                                              mds,
                                              modules,
                                              firstMDIndex,
                                              secondMDIndex,
                                              thirdMDIndex,
                                              fourthMDIndex,
                                              lowerModuleIndex1,
                                              lowerModuleIndex2,
                                              lowerModuleIndex3,
                                              lowerModuleIndex4,
                                              innerRadius,
                                              outerRadius,
                                              promptScore,
                                              displacedScore,
                                              fakeScore,
                                              regressionRadius,
                                              nonAnchorRegressionRadius,
                                              triplets.fakeScore()[innerTripletIndex],
                                              triplets.promptScore()[innerTripletIndex],
                                              triplets.displacedScore()[innerTripletIndex],
                                              triplets.fakeScore()[outerTripletIndex],
                                              triplets.promptScore()[outerTripletIndex],
                                              triplets.displacedScore()[outerTripletIndex]);

    if (!inference) {
      return false;
    }
    // only run dBeta selector for low/high pT to avoid removing displaced efficiency
    if (pt > 10 || pt < 1) {
      if (not runQuintupletdBetaAlgoSelector(acc,
                                             modules,
                                             mds,
                                             segments,
                                             lowerModuleIndex1,
                                             lowerModuleIndex2,
                                             lowerModuleIndex3,
                                             lowerModuleIndex4,
                                             firstSegmentIndex,
                                             thirdSegmentIndex,
                                             firstMDIndex,
                                             secondMDIndex,
                                             thirdMDIndex,
                                             fourthMDIndex,
                                             dBeta,
                                             ptCut))
        return false;
    } else {
      dBeta = 0;
    }

    if (not passT4RZConstraint(acc,
                               modules,
                               mds,
                               firstMDIndex,
                               secondMDIndex,
                               thirdMDIndex,
                               fourthMDIndex,
                               lowerModuleIndex1,
                               lowerModuleIndex2,
                               lowerModuleIndex3,
                               lowerModuleIndex4,
                               rzChiSquared,
                               inner_pt,
                               innerRadius,
                               inner_circleCenterX,
                               inner_circleCenterY,
                               innerT3charge))
      return false;

    float dxy = abs(std::hypot(regressionCenterX, regressionCenterY) - regressionRadius);
    float eta_layer3;
    const int layer1 = modules.layers()[lowerModuleIndex1];
    if (layer1 == 3) {
      eta_layer3 = alpaka::math::abs(acc, mds.anchorEta()[firstMDIndex]);
    } else if (layer1 == 2) {
      eta_layer3 = alpaka::math::abs(acc, mds.anchorEta()[secondMDIndex]);
    } else {
      eta_layer3 = alpaka::math::abs(acc, mds.anchorEta()[thirdMDIndex]);
    }
    if (dxy < 0.05f && eta_layer3 < 0.5f)
      return false;
    else if (dxy < 0.01f && eta_layer3 < 1.5f)
      return false;

    nonAnchorChiSquared = computeChiSquared(acc,
                                            Params_T4::kLayers,
                                            nonAnchorxs,
                                            nonAnchorys,
                                            nonAnchorDelta1,
                                            nonAnchorDelta2,
                                            nonAnchorSlopes,
                                            isFlat,
                                            regressionCenterX,
                                            regressionCenterY,
                                            regressionRadius);

    return true;
  };

  struct CreateQuadruplets {
    ALPAKA_FN_ACC void operator()(Acc3D const& acc,
                                  ModulesConst modules,
                                  MiniDoubletsConst mds,
                                  SegmentsConst segments,
                                  Triplets triplets,
                                  TripletsOccupancyConst tripletsOccupancy,
                                  Quadruplets quadruplets,
                                  QuadrupletsOccupancy quadrupletsOccupancy,
                                  ObjectRangesConst ranges,
                                  uint16_t nEligibleT4Modules,
                                  const float ptCut) const {
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[1] == 1) &&
                        (alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[2] == 1));

      unsigned int& matchCount = alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);

      const auto threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
      const auto blockDim = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);

      const int threadIdX = threadIdx.x();
      const int threadIdY = threadIdx.y();
      const int blockSizeX = blockDim.x();
      const int blockSizeY = blockDim.y();
      const int blockSize = blockSizeX * blockSizeY;
      const int flatThreadIdxXY = threadIdY * blockSizeX + threadIdX;
      const int flatThreadExtent = blockSize;  // total threads per block

      const auto& mdIndices = segments.mdIndices();
      const auto& segIdx = triplets.segmentIndices();
      const auto& lmIdx = triplets.lowerModuleIndices();
      const auto& tripIdx = ranges.tripletModuleIndices();

      for (int iter : cms::alpakatools::uniform_groups_z(acc, nEligibleT4Modules)) {
        const uint16_t lowerModule1 = ranges.indicesOfEligibleT4Modules()[iter];

        if (cms::alpakatools::once_per_block(acc)) {
          matchCount = 0;
        }

        short layer2_adjustment, md_adjustment;
        int layer = modules.layers()[lowerModule1];
        if (layer == 1) {
          if (modules.subdets()[lowerModule1] != Endcap)
            continue;
          layer2_adjustment = 1;
          md_adjustment = 1;
        }  // get upper segment to be in third layer
        else if (layer == 2) {
          if (modules.subdets()[lowerModule1] != Endcap)
            continue;
          layer2_adjustment = 1;
          md_adjustment = 0;
        }  // get lower segment to be in third layer
        else {
          layer2_adjustment = 0;
          md_adjustment = 0;
        }
        const unsigned int nInnerTriplets = tripletsOccupancy.nTriplets()[lowerModule1];

        alpaka::syncBlockThreads(acc);

        // Step 1: Make inner and outer triplet pairs
        for (unsigned int innerTripletArrayIndex : cms::alpakatools::uniform_elements_y(acc, nInnerTriplets)) {
          const unsigned int innerTripletIndex = tripIdx[lowerModule1] + innerTripletArrayIndex;
          if (triplets.partOfPT5()[innerTripletIndex])
            continue;  //don't create T4s for T3s accounted in pT5s
          if (triplets.partOfT5()[innerTripletIndex])
            continue;  //don't create T4s for T3s accounted in T5s
          if (triplets.partOfPT3()[innerTripletIndex])
            continue;  //don't create T4s for T3s accounted in pT3s
          const uint16_t lowerModule2 = lmIdx[innerTripletIndex][1];
          const unsigned int nOuterTriplets = tripletsOccupancy.nTriplets()[lowerModule2];
          for (unsigned int outerTripletArrayIndex : cms::alpakatools::uniform_elements_x(acc, nOuterTriplets)) {
            unsigned int outerTripletIndex = tripIdx[lowerModule2] + outerTripletArrayIndex;
            if (triplets.partOfPT5()[outerTripletIndex])
              continue;  //don't create T4s for T3s accounted in pT5s
            if (triplets.partOfT5()[outerTripletIndex])
              continue;  //don't create T4s for T3s accounted in T5s
            if (triplets.partOfPT3()[outerTripletIndex])
              continue;  //don't create T4s for T3s accounted in pT3s

            const unsigned int innerT3LS2Index = segIdx[innerTripletIndex][1];
            const unsigned int outerT3LS1Index = segIdx[outerTripletIndex][0];

            //check if the 2 T3s have a common LS
            if (innerT3LS2Index != outerT3LS1Index)
              continue;

            // If densely connected, do not attempt parallel processing to avoid truncation
            if (nInnerTriplets >= kNTripletThreshold || nOuterTriplets >= kNTripletThreshold) {
              const uint16_t lowerModule3 = lmIdx[outerTripletIndex][1];
              const uint16_t lowerModule4 = lmIdx[outerTripletIndex][2];

              float innerRadius = triplets.radius()[innerTripletIndex];
              float outerRadius = triplets.radius()[outerTripletIndex];
              float rzChiSquared, dBeta, nonAnchorChiSquared, regressionCenterX, regressionCenterY, regressionRadius,
                  nonAnchorRegressionRadius, chiSquared, promptScore, displacedScore, fakeScore;

              float pt = (innerRadius + outerRadius) * k2Rinv1GeVf;

              bool success = runQuadrupletDefaultAlgo(acc,
                                                      modules,
                                                      mds,
                                                      segments,
                                                      triplets,
                                                      lowerModule1,
                                                      lowerModule2,
                                                      lowerModule3,
                                                      lowerModule4,
                                                      innerTripletIndex,
                                                      outerTripletIndex,
                                                      regressionCenterX,
                                                      regressionCenterY,
                                                      regressionRadius,
                                                      nonAnchorRegressionRadius,
                                                      chiSquared,
                                                      ptCut,
                                                      rzChiSquared,
                                                      nonAnchorChiSquared,
                                                      dBeta,
                                                      promptScore,
                                                      displacedScore,
                                                      fakeScore);
              if (success) {
                int totOccupancyQuadruplets =
                    alpaka::atomicAdd(acc,
                                      &quadrupletsOccupancy.totOccupancyQuadruplets()[lowerModule1],
                                      1u,
                                      alpaka::hierarchy::Threads{});
                if (totOccupancyQuadruplets >= ranges.quadrupletModuleOccupancy()[lowerModule1]) {
#ifdef WARNINGS
                  printf("Quadruplet excess alert! Module index = %d, Occupancy = %d\n",
                         lowerModule1,
                         totOccupancyQuadruplets);
#endif
                } else {
                  int quadrupletModuleIndex = alpaka::atomicAdd(
                      acc, &quadrupletsOccupancy.nQuadruplets()[lowerModule1], 1u, alpaka::hierarchy::Threads{});
                  if (ranges.quadrupletModuleIndices()[lowerModule1] == -1) {
#ifdef WARNINGS
                    printf("Quadruplets : no memory for module at module index = %d\n", lowerModule1);
#endif
                  } else {
                    unsigned int quadrupletIndex =
                        ranges.quadrupletModuleIndices()[lowerModule1] + quadrupletModuleIndex;
                    const unsigned int layer3MDIndex =
                        mdIndices[segIdx[innerTripletIndex][md_adjustment]][layer2_adjustment];
                    float phi = mds.anchorPhi()[layer3MDIndex];
                    float eta = mds.anchorEta()[layer3MDIndex];

                    float scores = chiSquared + nonAnchorChiSquared;
                    addQuadrupletToMemory(triplets,
                                          quadruplets,
                                          innerTripletIndex,
                                          outerTripletIndex,
                                          lowerModule1,
                                          lowerModule2,
                                          lowerModule3,
                                          lowerModule4,
                                          innerRadius,
                                          outerRadius,
                                          pt,
                                          eta,
                                          phi,
                                          scores,
                                          layer,
                                          quadrupletIndex,
                                          rzChiSquared,
                                          dBeta,
                                          promptScore,
                                          displacedScore,
                                          fakeScore,
                                          regressionCenterX,
                                          regressionCenterY,
                                          regressionRadius,
                                          nonAnchorRegressionRadius);
                  }
                }
              }
              continue;
            }

            int mIdx = alpaka::atomicAdd(acc, &matchCount, 1u, alpaka::hierarchy::Threads{});
            const unsigned int quadrupletIndex = ranges.quadrupletModuleIndices()[lowerModule1] + mIdx;

#ifdef WARNINGS
            const unsigned int rightBound =
                static_cast<unsigned int>(ranges.quadrupletModuleIndices()[lowerModule1 + 1]);
            if (quadrupletIndex >= rightBound) {
              printf(
                  "Quadruplet module occupancy alert! module quadruplet starting index  = %d, Pair quadruplet index = "
                  "%d, next module quadruplet starting index = %d\n",
                  ranges.quadrupletModuleIndices()[lowerModule1],
                  mIdx,
                  ranges.quadrupletModuleIndices()[lowerModule1 + 1]);
            }
#endif

            quadruplets.preAllocatedTripletIndices()[quadrupletIndex][0] = innerTripletIndex;
            quadruplets.preAllocatedTripletIndices()[quadrupletIndex][1] = outerTripletIndex;
          }
        }

        alpaka::syncBlockThreads(acc);
        if (matchCount == 0) {
          continue;
        }

        // Step 2: Parallel processing of triplet pairs
        for (unsigned int i = flatThreadIdxXY; i < matchCount; i += flatThreadExtent) {
          const unsigned int quadrupletIndex = ranges.quadrupletModuleIndices()[lowerModule1] + i;
          const int innerTripletIndex = quadruplets.preAllocatedTripletIndices()[quadrupletIndex][0];
          const int outerTripletIndex = quadruplets.preAllocatedTripletIndices()[quadrupletIndex][1];

          const uint16_t lowerModule2 = lmIdx[innerTripletIndex][1];
          const uint16_t lowerModule3 = lmIdx[outerTripletIndex][1];
          const uint16_t lowerModule4 = lmIdx[outerTripletIndex][2];

          float innerRadius = triplets.radius()[innerTripletIndex];
          float outerRadius = triplets.radius()[outerTripletIndex];
          float rzChiSquared, dBeta, nonAnchorChiSquared, regressionCenterX, regressionCenterY, regressionRadius,
              nonAnchorRegressionRadius, chiSquared, promptScore, displacedScore, fakeScore;

          float pt = (innerRadius + outerRadius) * k2Rinv1GeVf;

          bool success = runQuadrupletDefaultAlgo(acc,
                                                  modules,
                                                  mds,
                                                  segments,
                                                  triplets,
                                                  lowerModule1,
                                                  lowerModule2,
                                                  lowerModule3,
                                                  lowerModule4,
                                                  innerTripletIndex,
                                                  outerTripletIndex,
                                                  regressionCenterX,
                                                  regressionCenterY,
                                                  regressionRadius,
                                                  nonAnchorRegressionRadius,
                                                  chiSquared,
                                                  ptCut,
                                                  rzChiSquared,
                                                  nonAnchorChiSquared,
                                                  dBeta,
                                                  promptScore,
                                                  displacedScore,
                                                  fakeScore);
          if (success) {
            int totOccupancyQuadruplets = alpaka::atomicAdd(
                acc, &quadrupletsOccupancy.totOccupancyQuadruplets()[lowerModule1], 1u, alpaka::hierarchy::Threads{});
            if (totOccupancyQuadruplets >= ranges.quadrupletModuleOccupancy()[lowerModule1]) {
#ifdef WARNINGS
              printf("Quadruplet excess alert! Module index = %d, Occupancy = %d\n",
                     lowerModule1,
                     totOccupancyQuadruplets);
#endif
            } else {
              int quadrupletModuleIndex = alpaka::atomicAdd(
                  acc, &quadrupletsOccupancy.nQuadruplets()[lowerModule1], 1u, alpaka::hierarchy::Threads{});
              if (ranges.quadrupletModuleIndices()[lowerModule1] == -1) {
#ifdef WARNINGS
                printf("Quadruplets : no memory for module at module index = %d\n", lowerModule1);
#endif
              } else {
                const unsigned int quadrupletIndex =
                    ranges.quadrupletModuleIndices()[lowerModule1] + quadrupletModuleIndex;
                const unsigned int layer3MDIndex =
                    mdIndices[segIdx[innerTripletIndex][md_adjustment]][layer2_adjustment];
                float phi = mds.anchorPhi()[layer3MDIndex];
                float eta = mds.anchorEta()[layer3MDIndex];

                float scores = chiSquared + nonAnchorChiSquared;
                addQuadrupletToMemory(triplets,
                                      quadruplets,
                                      innerTripletIndex,
                                      outerTripletIndex,
                                      lowerModule1,
                                      lowerModule2,
                                      lowerModule3,
                                      lowerModule4,
                                      innerRadius,
                                      outerRadius,
                                      pt,
                                      eta,
                                      phi,
                                      scores,
                                      layer,
                                      quadrupletIndex,
                                      rzChiSquared,
                                      dBeta,
                                      promptScore,
                                      displacedScore,
                                      fakeScore,
                                      regressionCenterX,
                                      regressionCenterY,
                                      regressionRadius,
                                      nonAnchorRegressionRadius);
              }
            }
          }
        }
      }
    }
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool isValidQuadRegion(ModulesConst modules, uint16_t lowerModule) {
    const short layer = modules.layers()[lowerModule];
    const short subdet = modules.subdets()[lowerModule];
    // Quadruplets starting outside these regions are not built.
    return (subdet == Barrel && layer > 2) || (subdet == Endcap);
  }

  struct CountTripletLSConnections {
    ALPAKA_FN_ACC void operator()(Acc3D const& acc,
                                  ModulesConst modules,
                                  MiniDoubletsConst mds,
                                  SegmentsConst segments,
                                  Triplets triplets,
                                  TripletsOccupancyConst tripletsOcc,
                                  ObjectRangesConst ranges,
                                  const float ptCut) const {
      // The atomicAdd below with hierarchy::Threads{} requires one block in x, y dimensions.
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[1] == 1) &&
                        (alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[2] == 1));
      const auto& mdIndices = segments.mdIndices();
      const auto& segIdx = triplets.segmentIndices();
      const auto& lmIdx = triplets.lowerModuleIndices();
      const auto& tripIdx = ranges.tripletModuleIndices();

      for (uint16_t lowerModule1 : cms::alpakatools::uniform_groups_z(acc, modules.nLowerModules())) {
        if (!isValidQuadRegion(modules, lowerModule1))
          continue;

        const unsigned int nInnerTriplets = tripletsOcc.nTriplets()[lowerModule1];
        if (nInnerTriplets == 0)
          continue;

        for (unsigned int innerTripletArrayIndex : cms::alpakatools::uniform_elements_y(acc, nInnerTriplets)) {
          const unsigned int innerTripletIndex = tripIdx[lowerModule1] + innerTripletArrayIndex;

          const uint16_t lowerModule2 = lmIdx[innerTripletIndex][1];
          const unsigned int nOuterTriplets = tripletsOcc.nTriplets()[lowerModule2];
          if (nOuterTriplets == 0)
            continue;

          const unsigned int secondSegIdx = segIdx[innerTripletIndex][1];
          const unsigned int secondMDInner = mdIndices[secondSegIdx][0];
          const unsigned int secondMDOuter = mdIndices[secondSegIdx][1];

          for (unsigned int outerTripletArrayIndex : cms::alpakatools::uniform_elements_x(acc, nOuterTriplets)) {
            const unsigned int outerTripletIndex = tripIdx[lowerModule2] + outerTripletArrayIndex;
            const unsigned int thirdSegIdx = segIdx[outerTripletIndex][0];
            const unsigned int thirdMDInner = mdIndices[thirdSegIdx][0];
            const unsigned int thirdMDOuter = mdIndices[thirdSegIdx][1];

            if ((secondMDInner == thirdMDInner) && (secondMDOuter == thirdMDOuter)) {
              // Will only perform runQuadrupletDefaultAlgorithm() checks if densely connected
              if (nInnerTriplets < kNTripletThreshold && nOuterTriplets < kNTripletThreshold) {
                alpaka::atomicAdd(acc, &triplets.connectedLSMax()[innerTripletIndex], 1u, alpaka::hierarchy::Threads{});
              } else {
                const uint16_t lowerModule3 = lmIdx[outerTripletIndex][1];
                const uint16_t lowerModule4 = lmIdx[outerTripletIndex][2];

                float rzChiSquared, dBeta, nonAnchorChiSquared, regressionCenterX, regressionCenterY, regressionRadius,
                    nonAnchorRegressionRadius, chiSquared, promptScore, displacedScore, fakeScore;

                const bool ok = runQuadrupletDefaultAlgo(acc,
                                                         modules,
                                                         mds,
                                                         segments,
                                                         triplets,
                                                         lowerModule1,
                                                         lowerModule2,
                                                         lowerModule3,
                                                         lowerModule4,
                                                         innerTripletIndex,
                                                         outerTripletIndex,
                                                         regressionCenterX,
                                                         regressionCenterY,
                                                         regressionRadius,
                                                         nonAnchorRegressionRadius,
                                                         chiSquared,
                                                         ptCut,
                                                         rzChiSquared,
                                                         nonAnchorChiSquared,
                                                         dBeta,
                                                         promptScore,
                                                         displacedScore,
                                                         fakeScore);
                if (ok) {
                  alpaka::atomicAdd(
                      acc, &triplets.connectedLSMax()[innerTripletIndex], 1u, alpaka::hierarchy::Threads{});
                }
              }
            }
          }
        }
      }
    }
  };

  struct CreateEligibleModulesListForQuadruplets {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  TripletsOccupancyConst tripletsOcc,
                                  ObjectRanges ranges,
                                  Triplets triplets) const {
      // Single-block kernel
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));

      int& nEligibleT4Modulesx = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      int& nTotalQuadrupletsx = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      if (cms::alpakatools::once_per_block(acc)) {
        nTotalQuadrupletsx = 0;
        nEligibleT4Modulesx = 0;
      }
      alpaka::syncBlockThreads(acc);

      for (uint16_t lowerModule : cms::alpakatools::uniform_elements(acc, modules.nLowerModules())) {
        if (!isValidQuadRegion(modules, lowerModule))
          continue;

        unsigned int nInnerTriplets = tripletsOcc.nTriplets()[lowerModule];
        if (nInnerTriplets == 0)
          continue;

        // Sum the real connectivity for triplets in this module
        int dynamic_count = 0;
        const unsigned int firstTripletIdx = ranges.tripletModuleIndices()[lowerModule];
        for (unsigned int t = 0; t < nInnerTriplets; ++t) {
          unsigned int tripletIndex = firstTripletIdx + t;
          dynamic_count += triplets.connectedLSMax()[tripletIndex];
        }

        if (dynamic_count == 0)
          continue;

        int nEligibleT4Modules = alpaka::atomicAdd(acc, &nEligibleT4Modulesx, 1, alpaka::hierarchy::Threads{});
        int nTotQ = alpaka::atomicAdd(acc, &nTotalQuadrupletsx, dynamic_count, alpaka::hierarchy::Threads{});

        ranges.quadrupletModuleIndices()[lowerModule] = nTotQ;
        ranges.indicesOfEligibleT4Modules()[nEligibleT4Modules] = lowerModule;
        ranges.quadrupletModuleOccupancy()[lowerModule] = dynamic_count;
      }

      // Wait for all threads to finish before reporting final values
      alpaka::syncBlockThreads(acc);
      if (cms::alpakatools::once_per_block(acc)) {
        ranges.nEligibleT4Modules() = static_cast<uint16_t>(nEligibleT4Modulesx);
        ranges.nTotalQuads() = static_cast<unsigned int>(nTotalQuadrupletsx);
      }
    }
  };

  struct AddQuadrupletRangesToEventExplicit {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  QuadrupletsOccupancyConst quadrupletsOccupancy,
                                  ObjectRanges ranges) const {
      // implementation is 1D with a single block
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));

      for (uint16_t i : cms::alpakatools::uniform_elements(acc, modules.nLowerModules())) {
        if (quadrupletsOccupancy.nQuadruplets()[i] == 0 or ranges.quadrupletModuleIndices()[i] == -1) {
          ranges.quadrupletRanges()[i][0] = -1;
          ranges.quadrupletRanges()[i][1] = -1;
        } else {
          ranges.quadrupletRanges()[i][0] = ranges.quadrupletModuleIndices()[i];
          ranges.quadrupletRanges()[i][1] =
              ranges.quadrupletModuleIndices()[i] + quadrupletsOccupancy.nQuadruplets()[i] - 1;
        }
      }
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
