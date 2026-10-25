#ifndef RecoTracker_LSTCore_src_alpaka_Quintuplet_h
#define RecoTracker_LSTCore_src_alpaka_Quintuplet_h

#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/warpsize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "FWCore/Utilities/interface/CMSUnrollLoop.h"

#include "RecoTracker/LSTCore/interface/ObjectRangesSoA.h"
#include "RecoTracker/LSTCore/interface/MiniDoubletsSoA.h"
#include "RecoTracker/LSTCore/interface/SegmentsSoA.h"
#include "RecoTracker/LSTCore/interface/TripletsSoA.h"
#include "RecoTracker/LSTCore/interface/QuintupletsSoA.h"
#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/ModulesSoA.h"
#include "RecoTracker/LSTCore/interface/EndcapGeometry.h"
#include "RecoTracker/LSTCore/interface/ObjectRangesSoA.h"
#include "RecoTracker/LSTCore/interface/Circle.h"

#include "NeuralNetwork.h"
#include "TripletAccessors.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addQuintupletToMemory(ModulesConst modules,
                                                            MiniDoubletsConst mds,
                                                            SegmentsConst segments,
                                                            TripletsConst triplets,
                                                            Quintuplets quintuplets,
                                                            QuintupletsByMD quintupletsByMD0,
                                                            QuintupletsByMD quintupletsByMD1,
                                                            unsigned int innerTripletIndex,
                                                            unsigned int outerTripletIndex,
                                                            uint16_t lowerModule1,
                                                            uint16_t lowerModule2,
                                                            uint16_t lowerModule3,
                                                            uint16_t lowerModule4,
                                                            uint16_t lowerModule5,
                                                            float innerRadius,
                                                            float bridgeRadius,
                                                            float regressionCenterX,
                                                            float regressionCenterY,
                                                            float regressionRadius,
                                                            float rzChiSquared,
                                                            float rPhiChiSquared,
                                                            float dBeta1,
                                                            float dBeta2,
                                                            float eta,
                                                            float phi,
                                                            uint8_t layer,
                                                            unsigned int quintupletIndex,
                                                            unsigned int quintupletByMD0Index,
                                                            unsigned int quintupletByMD1Index,
                                                            const float (&t5Embed)[Params_T5::kEmbed],
                                                            float dnnScore) {
    quintuplets.tripletIndices()[quintupletIndex][0] = innerTripletIndex;
    quintuplets.tripletIndices()[quintupletIndex][1] = outerTripletIndex;

    quintuplets.lowerModuleIndices()[quintupletIndex][0] = lowerModule1;
    quintuplets.lowerModuleIndices()[quintupletIndex][1] = lowerModule2;
    quintuplets.lowerModuleIndices()[quintupletIndex][2] = lowerModule3;
    quintuplets.lowerModuleIndices()[quintupletIndex][3] = lowerModule4;
    quintuplets.lowerModuleIndices()[quintupletIndex][4] = lowerModule5;
    quintuplets.innerRadius()[quintupletIndex] = __F2H(innerRadius);
    quintuplets.eta()[quintupletIndex] = __F2H(eta);
    quintuplets.phi()[quintupletIndex] = __F2H(phi);
    quintuplets.isDup()[quintupletIndex] = 0;
    quintuplets.nLayers()[quintupletIndex] = Params_T5::kBaseLayers;
    quintuplets.regressionRadius()[quintupletIndex] = regressionRadius;
    quintuplets.regressionCenterX()[quintupletIndex] = regressionCenterX;
    quintuplets.regressionCenterY()[quintupletIndex] = regressionCenterY;
    quintuplets.logicalLayers()[quintupletIndex][0] =
        getLogicalLayer(modules, triplets.lowerModuleIndices()[innerTripletIndex][0]);
    quintuplets.logicalLayers()[quintupletIndex][1] =
        getLogicalLayer(modules, triplets.lowerModuleIndices()[innerTripletIndex][1]);
    quintuplets.logicalLayers()[quintupletIndex][2] =
        getLogicalLayer(modules, triplets.lowerModuleIndices()[innerTripletIndex][2]);
    quintuplets.logicalLayers()[quintupletIndex][3] =
        getLogicalLayer(modules, triplets.lowerModuleIndices()[outerTripletIndex][1]);
    quintuplets.logicalLayers()[quintupletIndex][4] =
        getLogicalLayer(modules, triplets.lowerModuleIndices()[outerTripletIndex][2]);

    unsigned int innerT3Hits[Params_T3::kHits], outerT3Hits[Params_T3::kHits];
    getTripletHitIndices(mds, segments, triplets, innerTripletIndex, innerT3Hits);
    getTripletHitIndices(mds, segments, triplets, outerTripletIndex, outerT3Hits);

    quintuplets.hitIndices()[quintupletIndex][0] = innerT3Hits[0];
    quintuplets.hitIndices()[quintupletIndex][1] = innerT3Hits[1];
    quintuplets.hitIndices()[quintupletIndex][2] = innerT3Hits[2];
    quintuplets.hitIndices()[quintupletIndex][3] = innerT3Hits[3];
    quintuplets.hitIndices()[quintupletIndex][4] = innerT3Hits[4];
    quintuplets.hitIndices()[quintupletIndex][5] = innerT3Hits[5];
    quintuplets.hitIndices()[quintupletIndex][6] = outerT3Hits[2];
    quintuplets.hitIndices()[quintupletIndex][7] = outerT3Hits[3];
    quintuplets.hitIndices()[quintupletIndex][8] = outerT3Hits[4];
    quintuplets.hitIndices()[quintupletIndex][9] = outerT3Hits[5];
#ifdef CUT_VALUE_DEBUG
    quintuplets.bridgeRadius()[quintupletIndex] = bridgeRadius;
    quintuplets.rzChiSquared()[quintupletIndex] = rzChiSquared;
    quintuplets.chiSquared()[quintupletIndex] = rPhiChiSquared;
    quintuplets.dBeta1()[quintupletIndex] = dBeta1;
    quintuplets.dBeta2()[quintupletIndex] = dBeta2;
#endif
    quintuplets.dnnScore()[quintupletIndex] = dnnScore;

    CMS_UNROLL_LOOP
    for (unsigned int i = 0; i < Params_T5::kEmbed; ++i) {
      quintuplets.t5Embed()[quintupletIndex][i] = t5Embed[i];
    }

    // Initialize extended layer slots with sentinel values
    for (int i = Params_T5::kBaseLayers; i < Params_T5::kLayers; ++i) {
      quintuplets.logicalLayers()[quintupletIndex][i] = 0;
      quintuplets.lowerModuleIndices()[quintupletIndex][i] = lst::kTCEmptyLowerModule;
      quintuplets.hitIndices()[quintupletIndex][2 * i] = lst::kTCEmptyHitIdx;
      quintuplets.hitIndices()[quintupletIndex][2 * i + 1] = lst::kTCEmptyHitIdx;
    }

    auto const& lsIdx = triplets.segmentIndices();
    auto const& mdIdx = segments.mdIndices();
    const uint32_t md1Bar = mdIdx[lsIdx[innerTripletIndex][1]][0] & kT5ByMDBarCodeMask;
    const uint32_t md2Bar = mdIdx[lsIdx[innerTripletIndex][1]][1] & kT5ByMDBarCodeMask;
    const uint32_t md3Bar = mdIdx[lsIdx[outerTripletIndex][1]][0] & kT5ByMDBarCodeMask;
    const uint32_t md4Bar = mdIdx[lsIdx[outerTripletIndex][1]][1] & kT5ByMDBarCodeMask;
    const uint32_t md1234Bar =
        md1Bar | (md2Bar << kT5ByMDBarOffset) | (md3Bar << (kT5ByMDBarOffset * 2)) | (md4Bar << (kT5ByMDBarOffset * 3));
    if (quintupletByMD0Index != kInvalidU32Idx) {
      quintupletsByMD0.quintupletIndex()[quintupletByMD0Index] = quintupletIndex;
      quintupletsByMD0.mdBarCode()[quintupletByMD0Index] = md1234Bar;
    }
    if (quintupletByMD1Index != kInvalidU32Idx) {
      quintupletsByMD1.quintupletIndex()[quintupletByMD1Index] = quintupletIndex;
      //save the starting logical layer index instead of md1Bar
      quintupletsByMD1.mdBarCode()[quintupletByMD1Index] =
          (quintuplets.logicalLayers()[quintupletIndex][0] & kT5ByMDBarCodeMask) | (md1234Bar & ~kT5ByMDBarCodeMask);
    }
  }

  //bounds can be found at http://uaf-10.t2.ucsd.edu/~bsathian/SDL/T5_RZFix/t5_rz_thresholds.txt
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passT5RZConstraint(TAcc const& acc,
                                                         ModulesConst modules,
                                                         MiniDoubletsConst mds,
                                                         unsigned int firstMDIndex,
                                                         unsigned int secondMDIndex,
                                                         unsigned int thirdMDIndex,
                                                         unsigned int fourthMDIndex,
                                                         unsigned int fifthMDIndex,
                                                         uint16_t lowerModuleIndex1,
                                                         uint16_t lowerModuleIndex2,
                                                         uint16_t lowerModuleIndex3,
                                                         uint16_t lowerModuleIndex4,
                                                         uint16_t lowerModuleIndex5,
                                                         float& rzChiSquared,
                                                         float inner_pt,
                                                         float innerRadius,
                                                         float g,
                                                         float f) {
    //(g,f) is the center of the circle fitted by the innermost 3 points on x,y coordinates
    const float rt1 = mds.anchorRt()[firstMDIndex] / 100;  //in the unit of m instead of cm
    const float rt2 = mds.anchorRt()[secondMDIndex] / 100;
    const float rt3 = mds.anchorRt()[thirdMDIndex] / 100;
    const float rt4 = mds.anchorRt()[fourthMDIndex] / 100;
    const float rt5 = mds.anchorRt()[fifthMDIndex] / 100;

    const float z1 = mds.anchorZ()[firstMDIndex] / 100;
    const float z2 = mds.anchorZ()[secondMDIndex] / 100;
    const float z3 = mds.anchorZ()[thirdMDIndex] / 100;
    const float z4 = mds.anchorZ()[fourthMDIndex] / 100;
    const float z5 = mds.anchorZ()[fifthMDIndex] / 100;

    // Using lst_layer numbering convention defined in ModuleMethods.h
    const int layer1 = modules.lstLayers()[lowerModuleIndex1];
    const int layer2 = modules.lstLayers()[lowerModuleIndex2];
    const int layer3 = modules.lstLayers()[lowerModuleIndex3];
    const int layer4 = modules.lstLayers()[lowerModuleIndex4];
    const int layer5 = modules.lstLayers()[lowerModuleIndex5];

    //slope computed using the internal T3s
    const int moduleType1 = modules.moduleType()[lowerModuleIndex1];  //0 is ps, 1 is 2s
    const int moduleType2 = modules.moduleType()[lowerModuleIndex2];
    const int moduleType3 = modules.moduleType()[lowerModuleIndex3];
    const int moduleType4 = modules.moduleType()[lowerModuleIndex4];
    const int moduleType5 = modules.moduleType()[lowerModuleIndex5];

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
    float x_center = g / 100, y_center = f / 100;
    float x_init = mds.anchorX()[thirdMDIndex] / 100;
    float y_init = mds.anchorY()[thirdMDIndex] / 100;
    float z_init = mds.anchorZ()[thirdMDIndex] / 100;
    float rt_init = mds.anchorRt()[thirdMDIndex] / 100;  //use the second MD as initial point

    if (moduleType3 == 1)  // 1: if MD3 is in 2s layer
    {
      x_init = mds.anchorX()[secondMDIndex] / 100;
      y_init = mds.anchorY()[secondMDIndex] / 100;
      z_init = mds.anchorZ()[secondMDIndex] / 100;
      rt_init = mds.anchorRt()[secondMDIndex] / 100;
    }

    // start from a circle of inner T3.
    // to determine the charge
    int charge = 0;
    float slope3c = (y3 - y_center) / (x3 - x_center);
    float slope1c = (y1 - y_center) / (x1 - x_center);
    // these 4 "if"s basically separate the x-y plane into 4 quarters. It determines geometrically how a circle and line slope goes and their positions, and we can get the charges correspondingly.
    if ((y3 - y_center) > 0 && (y1 - y_center) > 0) {
      if (slope1c > 0 && slope3c < 0)
        charge = -1;  // on x axis of a quarter, 3 hits go anti-clockwise
      else if (slope1c < 0 && slope3c > 0)
        charge = 1;  // on x axis of a quarter, 3 hits go clockwise
      else if (slope3c > slope1c)
        charge = -1;
      else if (slope3c < slope1c)
        charge = 1;
    } else if ((y3 - y_center) < 0 && (y1 - y_center) < 0) {
      if (slope1c < 0 && slope3c > 0)
        charge = 1;
      else if (slope1c > 0 && slope3c < 0)
        charge = -1;
      else if (slope3c > slope1c)
        charge = -1;
      else if (slope3c < slope1c)
        charge = 1;
    } else if ((y3 - y_center) < 0 && (y1 - y_center) > 0) {
      if ((x3 - x_center) > 0 && (x1 - x_center) > 0)
        charge = 1;
      else if ((x3 - x_center) < 0 && (x1 - x_center) < 0)
        charge = -1;
    } else if ((y3 - y_center) > 0 && (y1 - y_center) < 0) {
      if ((x3 - x_center) > 0 && (x1 - x_center) > 0)
        charge = -1;
      else if ((x3 - x_center) < 0 && (x1 - x_center) < 0)
        charge = 1;
    }

    float pseudo_phi = alpaka::math::atan(
        acc, (y_init - y_center) / (x_init - x_center));  //actually represent pi/2-phi, wrt helix axis z
    float Pt = inner_pt, Px = Pt * alpaka::math::abs(acc, alpaka::math::sin(acc, pseudo_phi)),
          Py = Pt * alpaka::math::abs(acc, cos(pseudo_phi));

    // Above line only gives you the correct value of Px and Py, but signs of Px and Py calculated below.
    // We look at if the circle is clockwise or anti-clock wise, to make it simpler, we separate the x-y plane into 4 quarters.
    if (x_init > x_center && y_init > y_center)  //1st quad
    {
      if (charge == 1)
        Py = -Py;
      if (charge == -1)
        Px = -Px;
    }
    if (x_init < x_center && y_init > y_center)  //2nd quad
    {
      if (charge == -1) {
        Px = -Px;
        Py = -Py;
      }
    }
    if (x_init < x_center && y_init < y_center)  //3rd quad
    {
      if (charge == 1)
        Px = -Px;
      if (charge == -1)
        Py = -Py;
    }
    if (x_init > x_center && y_init < y_center)  //4th quad
    {
      if (charge == 1) {
        Px = -Px;
        Py = -Py;
      }
    }

    // But if the initial T5 curve goes across quarters(i.e. cross axis to separate the quarters), need special redeclaration of Px,Py signs on these to avoid errors
    if (moduleType3 == 0) {  // 0 is ps
      if (x4 < x3 && x3 < x2)
        Px = -alpaka::math::abs(acc, Px);
      else if (x4 > x3 && x3 > x2)
        Px = alpaka::math::abs(acc, Px);
      if (y4 < y3 && y3 < y2)
        Py = -alpaka::math::abs(acc, Py);
      else if (y4 > y3 && y3 > y2)
        Py = alpaka::math::abs(acc, Py);
    } else if (moduleType3 == 1)  // 1 is 2s
    {
      if (x3 < x2 && x2 < x1)
        Px = -alpaka::math::abs(acc, Px);
      else if (x3 > x2 && x2 > x1)
        Px = alpaka::math::abs(acc, Px);
      if (y3 < y2 && y2 < y1)
        Py = -alpaka::math::abs(acc, Py);
      else if (y3 > y2 && y2 > y1)
        Py = alpaka::math::abs(acc, Py);
    }

    //to get Pz, we use pt/pz=ds/dz, ds is the arclength between MD1 and MD3.
    float AO = alpaka::math::sqrt(acc, (x1 - x_center) * (x1 - x_center) + (y1 - y_center) * (y1 - y_center));
    float BO =
        alpaka::math::sqrt(acc, (x_init - x_center) * (x_init - x_center) + (y_init - y_center) * (y_init - y_center));
    float AB2 = (x1 - x_init) * (x1 - x_init) + (y1 - y_init) * (y1 - y_init);
    float dPhi = alpaka::math::acos(acc, (AO * AO + BO * BO - AB2) / (2 * AO * BO));
    float ds = innerRadius / 100 * dPhi;

    float Pz = (z_init - z1) / ds * Pt;
    float p = alpaka::math::sqrt(acc, Px * Px + Py * Py + Pz * Pz);

    float a = -2.f * k2Rinv1GeVf * 100 * charge;  // multiply by 100 to make the correct length units

    float zsi, rtsi;
    int layeri, moduleTypei;
    rzChiSquared = 0;
    for (size_t i = 2; i < 6; i++) {
      if (i == 2) {
        zsi = z2;
        rtsi = rt2;
        layeri = layer2;
        moduleTypei = moduleType2;
      } else if (i == 3) {
        zsi = z3;
        rtsi = rt3;
        layeri = layer3;
        moduleTypei = moduleType3;
      } else if (i == 4) {
        zsi = z4;
        rtsi = rt4;
        layeri = layer4;
        moduleTypei = moduleType4;
      } else if (i == 5) {
        zsi = z5;
        rtsi = rt5;
        layeri = layer5;
        moduleTypei = moduleType5;
      }

      if (moduleType3 == 0) {  //0: ps
        if (i == 3)
          continue;
      } else {
        if (i == 2)
          continue;
      }

      // calculation is copied from PixelTriplet.h computePT3RZChiSquared
      float diffr = 0, diffz = 0;

      float rou = a / p;
      // for endcap
      float s = (zsi - z_init) * p / Pz;
      float x = x_init + Px / a * alpaka::math::sin(acc, rou * s) - Py / a * (1 - alpaka::math::cos(acc, rou * s));
      float y = y_init + Py / a * alpaka::math::sin(acc, rou * s) + Px / a * (1 - alpaka::math::cos(acc, rou * s));
      diffr = (rtsi - alpaka::math::sqrt(acc, x * x + y * y)) * 100;

      // for barrel
      if (layeri <= 6) {
        float paraA =
            rt_init * rt_init + 2 * (Px * Px + Py * Py) / (a * a) + 2 * (y_init * Px - x_init * Py) / a - rtsi * rtsi;
        float paraB = 2 * (x_init * Px + y_init * Py) / a;
        float paraC = 2 * (y_init * Px - x_init * Py) / a + 2 * (Px * Px + Py * Py) / (a * a);
        float A = paraB * paraB + paraC * paraC;
        float B = 2 * paraA * paraB;
        float C = paraA * paraA - paraC * paraC;
        float sol1 = (-B + alpaka::math::sqrt(acc, B * B - 4 * A * C)) / (2 * A);
        float sol2 = (-B - alpaka::math::sqrt(acc, B * B - 4 * A * C)) / (2 * A);
        float solz1 = alpaka::math::asin(acc, sol1) / rou * Pz / p + z_init;
        float solz2 = alpaka::math::asin(acc, sol2) / rou * Pz / p + z_init;
        float diffz1 = (solz1 - zsi) * 100;
        float diffz2 = (solz2 - zsi) * 100;
        if (edm::isNotFinite(diffz1))
          diffz = diffz2;
        else if (edm::isNotFinite(diffz2))
          diffz = diffz1;
        else {
          diffz = (alpaka::math::abs(acc, diffz1) < alpaka::math::abs(acc, diffz2)) ? diffz1 : diffz2;
        }
      }
      residual = (layeri > 6) ? diffr : diffz;

      //PS Modules
      if (moduleTypei == 0) {
        error2 = kPixelPSZpitch * kPixelPSZpitch;
      } else  //2S modules
      {
        error2 = kStrip2SZpitch * kStrip2SZpitch;
      }

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
      if (i == 2 || i == 3) {
        residual = (layeri <= 6 && ((side == Center) or (drdz < 1))) ? diffz : diffr;
        float projection_missing2 = 1.f;
        if (drdz < 1)
          projection_missing2 =
              ((subdets == Endcap) or (side == Center)) ? 1.f : 1.f / (1 + drdz * drdz);  // cos(atan(drdz)), if dr/dz<1
        if (drdz > 1)
          projection_missing2 = ((subdets == Endcap) or (side == Center))
                                    ? 1.f
                                    : (drdz * drdz) / (1 + drdz * drdz);  //sin(atan(drdz)), if dr/dz>1
        error2 = error2 * projection_missing2;
      }
      rzChiSquared += 12 * (residual * residual) / error2;
    }
    // for set rzchi2 cut
    // if the 5 points are linear, helix calculation gives nan
    if (inner_pt > 100 || edm::isNotFinite(rzChiSquared)) {
      float slope;
      if (moduleType1 == 0 and moduleType2 == 0 and moduleType3 == 1)  //PSPS2S
      {
        slope = (z2 - z1) / (rt2 - rt1);
      } else {
        slope = (z3 - z1) / (rt3 - rt1);
      }
      float residual4_linear = (layer4 <= 6) ? ((z4 - z1) - slope * (rt4 - rt1)) : ((rt4 - rt1) - (z4 - z1) / slope);
      float residual5_linear = (layer4 <= 6) ? ((z5 - z1) - slope * (rt5 - rt1)) : ((rt5 - rt1) - (z5 - z1) / slope);

      // creating a chi squared type quantity
      // 0-> PS, 1->2S
      residual4_linear = (moduleType4 == 0) ? residual4_linear / kPixelPSZpitch : residual4_linear / kStrip2SZpitch;
      residual5_linear = (moduleType5 == 0) ? residual5_linear / kPixelPSZpitch : residual5_linear / kStrip2SZpitch;
      residual4_linear = residual4_linear * 100;
      residual5_linear = residual5_linear * 100;

      rzChiSquared = 12 * (residual4_linear * residual4_linear + residual5_linear * residual5_linear);
      return rzChiSquared < 4.677f;
    }

    // The category numbers are related to module regions and layers, decoding of the region numbers can be found here in slide 2 table. https://github.com/SegmentLinking/TrackLooper/files/11420927/part.2.pdf
    // The commented numbers after each case is the region code, and can look it up from the table to see which category it belongs to. For example, //0 means T5 built with Endcap 1,2,3,4,5 ps modules
    if (layer1 == 7 and layer2 == 8 and layer3 == 9 and layer4 == 10 and layer5 == 11)  //0
    {
      return true;
    } else if (layer1 == 7 and layer2 == 8 and layer3 == 9 and layer4 == 10 and layer5 == 16)  //1
    {
      return rzChiSquared < 37.956f;
    } else if (layer1 == 7 and layer2 == 8 and layer3 == 9 and layer4 == 15 and layer5 == 16)  //2
    {
      return rzChiSquared < 11.622f;
    } else if (layer1 == 1 and layer2 == 7 and layer3 == 8 and layer4 == 9) {
      if (layer5 == 10)  //3
      {
        return true;
      }
      if (layer5 == 15)  //4
      {
        return rzChiSquared < 37.941f;
      }
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 7) {
      if (layer4 == 8 and layer5 == 9)  //5
      {
        return true;
      }
      if (layer4 == 8 and layer5 == 14)  //6
      {
        return rzChiSquared < 52.561f;
      } else if (layer4 == 13 and layer5 == 14)  //7
      {
        return rzChiSquared < 13.76f;
      }
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 3) {
      if (layer4 == 7 and layer5 == 8)  //8
      {
        return rzChiSquared < 44.247f;
      } else if (layer4 == 7 and layer5 == 13)  //9
      {
        return rzChiSquared < 33.752f;
      } else if (layer4 == 12 and layer5 == 13)  //10
      {
        return rzChiSquared < 21.213f;
      } else if (layer4 == 4 and layer5 == 5)  //11
      {
        return rzChiSquared < 29.035f;
      } else if (layer4 == 4 and layer5 == 12)  //12
      {
        return rzChiSquared < 23.037f;
      }
    } else if (layer1 == 2 and layer2 == 7 and layer3 == 8) {
      if (layer4 == 9 and layer5 == 15)  //14
      {
        return rzChiSquared < 41.036f;
      } else if (layer4 == 14 and layer5 == 15)  //15
      {
        return rzChiSquared < 14.092f;
      }
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 7) {
      if (layer4 == 8 and layer5 == 14)  //16
      {
        return rzChiSquared < 23.748f;
      }
      if (layer4 == 13 and layer5 == 14)  //17
      {
        return rzChiSquared < 17.945f;
      }
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 4) {
      if (layer4 == 5 and layer5 == 6)  //18
      {
        return rzChiSquared < 8.803f;
      } else if (layer4 == 5 and layer5 == 12)  //19
      {
        return rzChiSquared < 7.930f;
      }

      else if (layer4 == 12 and layer5 == 13)  //20
      {
        return rzChiSquared < 7.626f;
      }
    }
    return true;
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool T5HasCommonMiniDoublet(TripletsConst triplets,
                                                             SegmentsConst segments,
                                                             unsigned int innerTripletIndex,
                                                             unsigned int outerTripletIndex) {
    unsigned int innerOuterSegmentIndex = triplets.segmentIndices()[innerTripletIndex][1];
    unsigned int outerInnerSegmentIndex = triplets.segmentIndices()[outerTripletIndex][0];
    unsigned int innerOuterOuterMiniDoubletIndex =
        segments.mdIndices()[innerOuterSegmentIndex][1];  //inner triplet outer segment outer MD index
    unsigned int outerInnerInnerMiniDoubletIndex =
        segments.mdIndices()[outerInnerSegmentIndex][0];  //outer triplet inner segment inner MD index

    return (innerOuterOuterMiniDoubletIndex == outerInnerInnerMiniDoubletIndex);
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void computeSigmasForRegression(TAcc const& acc,
                                                                 ModulesConst modules,
                                                                 const uint16_t* lowerModuleIndices,
                                                                 float* delta1,
                                                                 float* delta2,
                                                                 float* slopes,
                                                                 bool* isFlat,
                                                                 unsigned int nPoints = 5,
                                                                 bool anchorHits = true) {
    /*
    Bool anchorHits required to deal with a weird edge case wherein 
    the hits ultimately used in the regression are anchor hits, but the
    lower modules need not all be Pixel Modules (in case of PS). Similarly,
    when we compute the chi squared for the non-anchor hits, the "partner module"
    need not always be a PS strip module, but all non-anchor hits sit on strip 
    modules.
    */

    ModuleType moduleType;
    short moduleSubdet, moduleSide;
    float inv1 = kWidthPS / kWidth2S;
    float inv2 = kPixelPSZpitch / kWidth2S;
    float inv3 = kStripPSZpitch / kWidth2S;
    for (size_t i = 0; i < nPoints; i++) {
      moduleType = modules.moduleType()[lowerModuleIndices[i]];
      moduleSubdet = modules.subdets()[lowerModuleIndices[i]];
      moduleSide = modules.sides()[lowerModuleIndices[i]];
      const float& drdz = modules.drdzs()[lowerModuleIndices[i]];
      slopes[i] = modules.dxdys()[lowerModuleIndices[i]];
      //category 1 - barrel PS flat
      if (moduleSubdet == Barrel and moduleType == PS and moduleSide == Center) {
        delta1[i] = inv1;
        delta2[i] = inv1;
        slopes[i] = -999.f;
        isFlat[i] = true;
      }
      //category 2 - barrel 2S
      else if (moduleSubdet == Barrel and moduleType == TwoS) {
        delta1[i] = 1.f;
        delta2[i] = 1.f;
        slopes[i] = -999.f;
        isFlat[i] = true;
      }
      //category 3 - barrel PS tilted
      else if (moduleSubdet == Barrel and moduleType == PS and moduleSide != Center) {
        delta1[i] = inv1;
        isFlat[i] = false;

        if (anchorHits) {
          delta2[i] = (inv2 * drdz / alpaka::math::sqrt(acc, 1 + drdz * drdz));
        } else {
          delta2[i] = (inv3 * drdz / alpaka::math::sqrt(acc, 1 + drdz * drdz));
        }
      }
      //category 4 - endcap PS
      else if (moduleSubdet == Endcap and moduleType == PS) {
        delta1[i] = inv1;
        isFlat[i] = false;

        /*
        despite the type of the module layer of the lower module index,
        all anchor hits are on the pixel side and all non-anchor hits are
        on the strip side!
        */
        if (anchorHits) {
          delta2[i] = inv2;
        } else {
          delta2[i] = inv3;
        }
      }
      //category 5 - endcap 2S
      else if (moduleSubdet == Endcap and moduleType == TwoS) {
        delta1[i] = 1.f;
        delta2[i] = 500.f * inv1;
        isFlat[i] = false;
      } else {
#ifdef WARNINGS
        printf("ERROR!!!!! I SHOULDN'T BE HERE!!!! subdet = %d, type = %d, side = %d\n",
               moduleSubdet,
               moduleType,
               moduleSide);
#endif
      }
    }
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computeRadiusUsingRegression(TAcc const& acc,
                                                                    unsigned int nPoints,
                                                                    float* xs,
                                                                    float* ys,
                                                                    float* delta1,
                                                                    float* delta2,
                                                                    float* slopes,
                                                                    bool* isFlat,
                                                                    float& g,
                                                                    float& f,
                                                                    float* sigmas2,
                                                                    float& chiSquared) {
    float radius = 0.f;

    // Some extra variables
    // the two variables will be called x1 and x2, and y (which is x^2 + y^2)

    float sigmaX1Squared = 0.f;
    float sigmaX2Squared = 0.f;
    float sigmaX1X2 = 0.f;
    float sigmaX1y = 0.f;
    float sigmaX2y = 0.f;
    float sigmaY = 0.f;
    float sigmaX1 = 0.f;
    float sigmaX2 = 0.f;
    float sigmaOne = 0.f;

    float xPrime, yPrime, absArctanSlope, angleM;
    for (size_t i = 0; i < nPoints; i++) {
      // Computing sigmas is a very tricky affair
      // if the module is tilted or endcap, we need to use the slopes properly!

      absArctanSlope =
          (edm::isFinite(slopes[i]) ? alpaka::math::abs(acc, alpaka::math::atan(acc, slopes[i])) : kPi / 2.f);

      if (xs[i] > 0 and ys[i] > 0) {
        angleM = kPi / 2.f - absArctanSlope;
      } else if (xs[i] < 0 and ys[i] > 0) {
        angleM = absArctanSlope + kPi / 2.f;
      } else if (xs[i] < 0 and ys[i] < 0) {
        angleM = -(absArctanSlope + kPi / 2.f);
      } else if (xs[i] > 0 and ys[i] < 0) {
        angleM = -(kPi / 2.f - absArctanSlope);
      } else {
        angleM = 0;
      }

      if (not isFlat[i]) {
        xPrime = xs[i] * alpaka::math::cos(acc, angleM) + ys[i] * alpaka::math::sin(acc, angleM);
        yPrime = ys[i] * alpaka::math::cos(acc, angleM) - xs[i] * alpaka::math::sin(acc, angleM);
      } else {
        xPrime = xs[i];
        yPrime = ys[i];
      }
      sigmas2[i] = 4 * ((xPrime * delta1[i]) * (xPrime * delta1[i]) + (yPrime * delta2[i]) * (yPrime * delta2[i]));

      sigmaX1Squared += (xs[i] * xs[i]) / sigmas2[i];
      sigmaX2Squared += (ys[i] * ys[i]) / sigmas2[i];
      sigmaX1X2 += (xs[i] * ys[i]) / sigmas2[i];
      sigmaX1y += (xs[i] * (xs[i] * xs[i] + ys[i] * ys[i])) / sigmas2[i];
      sigmaX2y += (ys[i] * (xs[i] * xs[i] + ys[i] * ys[i])) / sigmas2[i];
      sigmaY += (xs[i] * xs[i] + ys[i] * ys[i]) / sigmas2[i];
      sigmaX1 += xs[i] / sigmas2[i];
      sigmaX2 += ys[i] / sigmas2[i];
      sigmaOne += 1.0f / sigmas2[i];
    }
    float denominator = (sigmaX1X2 - sigmaX1 * sigmaX2) * (sigmaX1X2 - sigmaX1 * sigmaX2) -
                        (sigmaX1Squared - sigmaX1 * sigmaX1) * (sigmaX2Squared - sigmaX2 * sigmaX2);

    float twoG = ((sigmaX2y - sigmaX2 * sigmaY) * (sigmaX1X2 - sigmaX1 * sigmaX2) -
                  (sigmaX1y - sigmaX1 * sigmaY) * (sigmaX2Squared - sigmaX2 * sigmaX2)) /
                 denominator;
    float twoF = ((sigmaX1y - sigmaX1 * sigmaY) * (sigmaX1X2 - sigmaX1 * sigmaX2) -
                  (sigmaX2y - sigmaX2 * sigmaY) * (sigmaX1Squared - sigmaX1 * sigmaX1)) /
                 denominator;

    float c = -(sigmaY - twoG * sigmaX1 - twoF * sigmaX2) / sigmaOne;
    g = 0.5f * twoG;
    f = 0.5f * twoF;
    if (g * g + f * f - c < 0) {
#ifdef WARNINGS
      printf("FATAL! r^2 < 0!\n");
#endif
      chiSquared = -1;
      return -1;
    }

    radius = alpaka::math::sqrt(acc, g * g + f * f - c);
    // compute chi squared
    chiSquared = 0.f;
    for (size_t i = 0; i < nPoints; i++) {
      chiSquared += (xs[i] * xs[i] + ys[i] * ys[i] - twoG * xs[i] - twoF * ys[i] + c) *
                    (xs[i] * xs[i] + ys[i] * ys[i] - twoG * xs[i] - twoF * ys[i] + c) / sigmas2[i];
    }
    return radius;
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computeChiSquared(TAcc const& acc,
                                                         unsigned int nPoints,
                                                         float* xs,
                                                         float* ys,
                                                         float* delta1,
                                                         float* delta2,
                                                         float* slopes,
                                                         bool* isFlat,
                                                         float g,
                                                         float f,
                                                         float radius) {
    // given values of (g, f, radius) and a set of points (and its uncertainties)
    // compute chi squared
    float c = g * g + f * f - radius * radius;
    float chiSquared = 0.f;
    float absArctanSlope, angleM, xPrime, yPrime, sigma2;
    for (size_t i = 0; i < nPoints; i++) {
      absArctanSlope =
          (edm::isFinite(slopes[i]) ? alpaka::math::abs(acc, alpaka::math::atan(acc, slopes[i])) : kPi / 2.f);
      if (xs[i] > 0 and ys[i] > 0) {
        angleM = kPi / 2.f - absArctanSlope;
      } else if (xs[i] < 0 and ys[i] > 0) {
        angleM = absArctanSlope + kPi / 2.f;
      } else if (xs[i] < 0 and ys[i] < 0) {
        angleM = -(absArctanSlope + kPi / 2.f);
      } else if (xs[i] > 0 and ys[i] < 0) {
        angleM = -(kPi / 2.f - absArctanSlope);
      } else {
        angleM = 0;
      }

      if (not isFlat[i]) {
        xPrime = xs[i] * alpaka::math::cos(acc, angleM) + ys[i] * alpaka::math::sin(acc, angleM);
        yPrime = ys[i] * alpaka::math::cos(acc, angleM) - xs[i] * alpaka::math::sin(acc, angleM);
      } else {
        xPrime = xs[i];
        yPrime = ys[i];
      }
      sigma2 = 4 * ((xPrime * delta1[i]) * (xPrime * delta1[i]) + (yPrime * delta2[i]) * (yPrime * delta2[i]));
      chiSquared += (xs[i] * xs[i] + ys[i] * ys[i] - 2 * g * xs[i] - 2 * f * ys[i] + c) *
                    (xs[i] * xs[i] + ys[i] * ys[i] - 2 * g * xs[i] - 2 * f * ys[i] + c) / sigma2;
    }
    return chiSquared;
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void runDeltaBetaIterations(TAcc const& acc,
                                                             float& betaIn,
                                                             float& betaOut,
                                                             float& pt_beta,
                                                             float sdIn_dr,
                                                             float sdOut_dr,
                                                             float dr,
                                                             float lIn,
                                                             bool useBetaInSign = false) {
    if (lIn == 0) {
      betaOut += alpaka::math::copysign(
          acc,
          alpaka::math::asin(
              acc, alpaka::math::min(acc, sdOut_dr * k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), kSinAlphaMax)),
          useBetaInSign ? betaIn : betaOut);
      return;
    }

    if (betaIn * betaOut > 0.f and
        (alpaka::math::abs(acc, pt_beta) < 4.f * kPt_betaMax or
         (lIn >= 11 and alpaka::math::abs(acc, pt_beta) <
                            8.f * kPt_betaMax)))  //and the pt_beta is well-defined; less strict for endcap-endcap
    {
      const float betaInUpd =
          betaIn +
          alpaka::math::copysign(
              acc,
              alpaka::math::asin(
                  acc, alpaka::math::min(acc, sdIn_dr * k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), kSinAlphaMax)),
              betaIn);  //FIXME: need a faster version
      const float betaOutUpd =
          betaOut +
          alpaka::math::copysign(
              acc,
              alpaka::math::asin(
                  acc, alpaka::math::min(acc, sdOut_dr * k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), kSinAlphaMax)),
              betaOut);  //FIXME: need a faster version
      float betaAv = 0.5f * (betaInUpd + betaOutUpd);

      //1st update
      const float pt_beta_inv =
          1.f / alpaka::math::abs(acc, dr * k2Rinv1GeVf / alpaka::math::sin(acc, betaAv));  //get a better pt estimate

      betaIn += alpaka::math::copysign(
          acc,
          alpaka::math::asin(acc, alpaka::math::min(acc, sdIn_dr * k2Rinv1GeVf * pt_beta_inv, kSinAlphaMax)),
          betaIn);  //FIXME: need a faster version
      betaOut += alpaka::math::copysign(
          acc,
          alpaka::math::asin(acc, alpaka::math::min(acc, sdOut_dr * k2Rinv1GeVf * pt_beta_inv, kSinAlphaMax)),
          betaOut);  //FIXME: need a faster version
      //update the av and pt
      betaAv = 0.5f * (betaIn + betaOut);
      //2nd update
      pt_beta = dr * k2Rinv1GeVf / alpaka::math::sin(acc, betaAv);  //get a better pt estimate
    } else if (lIn < 11 && alpaka::math::abs(acc, betaOut) < 0.2f * alpaka::math::abs(acc, betaIn) &&
               alpaka::math::abs(acc, pt_beta) < 12.f * kPt_betaMax)  //use betaIn sign as ref
    {
      const float pt_betaIn = dr * k2Rinv1GeVf / alpaka::math::sin(acc, betaIn);

      const float betaInUpd =
          betaIn +
          alpaka::math::copysign(
              acc,
              alpaka::math::asin(
                  acc, alpaka::math::min(acc, sdIn_dr * k2Rinv1GeVf / alpaka::math::abs(acc, pt_betaIn), kSinAlphaMax)),
              betaIn);  //FIXME: need a faster version
      const float betaOutUpd =
          betaOut +
          alpaka::math::copysign(
              acc,
              alpaka::math::asin(
                  acc,
                  alpaka::math::min(acc, sdOut_dr * k2Rinv1GeVf / alpaka::math::abs(acc, pt_betaIn), kSinAlphaMax)),
              betaIn);  //FIXME: need a faster version
      float betaAv = (alpaka::math::abs(acc, betaOut) > 0.2f * alpaka::math::abs(acc, betaIn))
                         ? (0.5f * (betaInUpd + betaOutUpd))
                         : betaInUpd;

      //1st update
      pt_beta = dr * k2Rinv1GeVf / alpaka::math::sin(acc, betaAv);  //get a better pt estimate
      betaIn += alpaka::math::copysign(
          acc,
          alpaka::math::asin(
              acc, alpaka::math::min(acc, sdIn_dr * k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), kSinAlphaMax)),
          betaIn);  //FIXME: need a faster version
      betaOut += alpaka::math::copysign(
          acc,
          alpaka::math::asin(
              acc, alpaka::math::min(acc, sdOut_dr * k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), kSinAlphaMax)),
          betaIn);  //FIXME: need a faster version
      //update the av and pt
      betaAv = 0.5f * (betaIn + betaOut);
      //2nd update
      pt_beta = dr * k2Rinv1GeVf / alpaka::math::sin(acc, betaAv);  //get a better pt estimate
    }
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuintupletdBetaCutBBBB(TAcc const& acc,
                                                                ModulesConst modules,
                                                                MiniDoubletsConst mds,
                                                                SegmentsConst segments,
                                                                uint16_t innerInnerLowerModuleIndex,
                                                                uint16_t innerOuterLowerModuleIndex,
                                                                uint16_t outerInnerLowerModuleIndex,
                                                                uint16_t outerOuterLowerModuleIndex,
                                                                unsigned int innerSegmentIndex,
                                                                unsigned int outerSegmentIndex,
                                                                unsigned int firstMDIndex,
                                                                unsigned int secondMDIndex,
                                                                unsigned int thirdMDIndex,
                                                                unsigned int fourthMDIndex,
                                                                float& dBeta,
                                                                const float ptCut) {
    float rt_InLo = mds.anchorRt()[firstMDIndex];
    float rt_InOut = mds.anchorRt()[secondMDIndex];
    float rt_OutLo = mds.anchorRt()[thirdMDIndex];

    float z_InLo = mds.anchorZ()[firstMDIndex];
    float z_OutLo = mds.anchorZ()[thirdMDIndex];

    float r3_InLo = alpaka::math::sqrt(acc, z_InLo * z_InLo + rt_InLo * rt_InLo);
    float drt_InSeg = rt_InOut - rt_InLo;

    float thetaMuls2 = (kMulsInGeV * kMulsInGeV) * (0.1f + 0.2f * (rt_OutLo - rt_InLo) / 50.f) * (r3_InLo / rt_InLo);

    float midPointX = 0.5f * (mds.anchorX()[firstMDIndex] + mds.anchorX()[thirdMDIndex]);
    float midPointY = 0.5f * (mds.anchorY()[firstMDIndex] + mds.anchorY()[thirdMDIndex]);
    float diffX = mds.anchorX()[thirdMDIndex] - mds.anchorX()[firstMDIndex];
    float diffY = mds.anchorY()[thirdMDIndex] - mds.anchorY()[firstMDIndex];

    float dPhi = cms::alpakatools::deltaPhi(acc, midPointX, midPointY, diffX, diffY);

    // First obtaining the raw betaIn and betaOut values without any correction and just purely based on the mini-doublet hit positions
    float alpha_InLo = __H2F(segments.dPhiChanges()[innerSegmentIndex]);
    float alpha_OutLo = __H2F(segments.dPhiChanges()[outerSegmentIndex]);

    bool isEC_lastLayer = modules.subdets()[outerOuterLowerModuleIndex] == Endcap and
                          modules.moduleType()[outerOuterLowerModuleIndex] == TwoS;

    float alpha_OutUp, alpha_OutUp_highEdge, alpha_OutUp_lowEdge;

    alpha_OutUp = cms::alpakatools::reducePhiRange(
        acc,
        cms::alpakatools::phi(acc,
                              mds.anchorX()[fourthMDIndex] - mds.anchorX()[thirdMDIndex],
                              mds.anchorY()[fourthMDIndex] - mds.anchorY()[thirdMDIndex]) -
            mds.anchorPhi()[fourthMDIndex]);

    alpha_OutUp_highEdge = alpha_OutUp;
    alpha_OutUp_lowEdge = alpha_OutUp;

    float tl_axis_x = mds.anchorX()[fourthMDIndex] - mds.anchorX()[firstMDIndex];
    float tl_axis_y = mds.anchorY()[fourthMDIndex] - mds.anchorY()[firstMDIndex];
    float tl_axis_highEdge_x = tl_axis_x;
    float tl_axis_highEdge_y = tl_axis_y;
    float tl_axis_lowEdge_x = tl_axis_x;
    float tl_axis_lowEdge_y = tl_axis_y;

    float betaIn =
        alpha_InLo - cms::alpakatools::reducePhiRange(
                         acc, cms::alpakatools::phi(acc, tl_axis_x, tl_axis_y) - mds.anchorPhi()[firstMDIndex]);

    float betaInRHmin = betaIn;
    float betaInRHmax = betaIn;
    float betaOut =
        -alpha_OutUp + cms::alpakatools::reducePhiRange(
                           acc, cms::alpakatools::phi(acc, tl_axis_x, tl_axis_y) - mds.anchorPhi()[fourthMDIndex]);

    float betaOutRHmin = betaOut;
    float betaOutRHmax = betaOut;

    // outer MD strip edges (2S endcap only): anchor xy +- the module's strip half-vector
    float highEdgeX_OutUp = 0.f, highEdgeY_OutUp = 0.f, lowEdgeX_OutUp = 0.f, lowEdgeY_OutUp = 0.f;
    if (isEC_lastLayer) {
      highEdgeX_OutUp = mds.anchorX()[fourthMDIndex] + modules.edgeDx()[outerOuterLowerModuleIndex];
      highEdgeY_OutUp = mds.anchorY()[fourthMDIndex] + modules.edgeDy()[outerOuterLowerModuleIndex];
      lowEdgeX_OutUp = mds.anchorX()[fourthMDIndex] - modules.edgeDx()[outerOuterLowerModuleIndex];
      lowEdgeY_OutUp = mds.anchorY()[fourthMDIndex] - modules.edgeDy()[outerOuterLowerModuleIndex];
      const float highEdgePhi_OutUp = alpaka::math::atan2(acc, highEdgeY_OutUp, highEdgeX_OutUp);
      const float lowEdgePhi_OutUp = alpaka::math::atan2(acc, lowEdgeY_OutUp, lowEdgeX_OutUp);
      alpha_OutUp_highEdge = cms::alpakatools::reducePhiRange(
          acc,
          cms::alpakatools::phi(
              acc, highEdgeX_OutUp - mds.anchorX()[thirdMDIndex], highEdgeY_OutUp - mds.anchorY()[thirdMDIndex]) -
              highEdgePhi_OutUp);
      alpha_OutUp_lowEdge = cms::alpakatools::reducePhiRange(
          acc,
          cms::alpakatools::phi(
              acc, lowEdgeX_OutUp - mds.anchorX()[thirdMDIndex], lowEdgeY_OutUp - mds.anchorY()[thirdMDIndex]) -
              lowEdgePhi_OutUp);

      tl_axis_highEdge_x = highEdgeX_OutUp - mds.anchorX()[firstMDIndex];
      tl_axis_highEdge_y = highEdgeY_OutUp - mds.anchorY()[firstMDIndex];
      tl_axis_lowEdge_x = lowEdgeX_OutUp - mds.anchorX()[firstMDIndex];
      tl_axis_lowEdge_y = lowEdgeY_OutUp - mds.anchorY()[firstMDIndex];

      betaOutRHmin = -alpha_OutUp_highEdge +
                     cms::alpakatools::reducePhiRange(
                         acc, cms::alpakatools::phi(acc, tl_axis_highEdge_x, tl_axis_highEdge_y) - highEdgePhi_OutUp);
      betaOutRHmax = -alpha_OutUp_lowEdge +
                     cms::alpakatools::reducePhiRange(
                         acc, cms::alpakatools::phi(acc, tl_axis_lowEdge_x, tl_axis_lowEdge_y) - lowEdgePhi_OutUp);
    }

    //beta computation
    float drt_tl_axis = alpaka::math::sqrt(acc, tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);

    //innerOuterAnchor - innerInnerAnchor
    const float rt_InSeg = alpaka::math::sqrt(acc,
                                              (mds.anchorX()[secondMDIndex] - mds.anchorX()[firstMDIndex]) *
                                                      (mds.anchorX()[secondMDIndex] - mds.anchorX()[firstMDIndex]) +
                                                  (mds.anchorY()[secondMDIndex] - mds.anchorY()[firstMDIndex]) *
                                                      (mds.anchorY()[secondMDIndex] - mds.anchorY()[firstMDIndex]));

    float betaAv = 0.5f * (betaIn + betaOut);
    float pt_beta = drt_tl_axis * k2Rinv1GeVf / alpaka::math::sin(acc, betaAv);
    int lIn = 5;
    int lOut = isEC_lastLayer ? 11 : 5;
    float sdOut_dr = alpaka::math::sqrt(acc,
                                        (mds.anchorX()[fourthMDIndex] - mds.anchorX()[thirdMDIndex]) *
                                                (mds.anchorX()[fourthMDIndex] - mds.anchorX()[thirdMDIndex]) +
                                            (mds.anchorY()[fourthMDIndex] - mds.anchorY()[thirdMDIndex]) *
                                                (mds.anchorY()[fourthMDIndex] - mds.anchorY()[thirdMDIndex]));
    float sdOut_d = mds.anchorRt()[fourthMDIndex] - mds.anchorRt()[thirdMDIndex];

    runDeltaBetaIterations(acc, betaIn, betaOut, pt_beta, rt_InSeg, sdOut_dr, drt_tl_axis, lIn);

    const float betaInMMSF = (alpaka::math::abs(acc, betaInRHmin + betaInRHmax) > 0)
                                 ? (2.f * betaIn / alpaka::math::abs(acc, betaInRHmin + betaInRHmax))
                                 : 0.f;  //mean value of min,max is the old betaIn
    const float betaOutMMSF = (alpaka::math::abs(acc, betaOutRHmin + betaOutRHmax) > 0)
                                  ? (2.f * betaOut / alpaka::math::abs(acc, betaOutRHmin + betaOutRHmax))
                                  : 0.f;
    betaInRHmin *= betaInMMSF;
    betaInRHmax *= betaInMMSF;
    betaOutRHmin *= betaOutMMSF;
    betaOutRHmax *= betaOutMMSF;

    float min_ptBeta_maxPtBeta = alpaka::math::min(
        acc, alpaka::math::abs(acc, pt_beta), kPt_betaMax);  //need to confimm the range-out value of 7 GeV
    const float dBetaMuls2 = thetaMuls2 * 16.f / (min_ptBeta_maxPtBeta * min_ptBeta_maxPtBeta);

    const float alphaInAbsReg =
        alpaka::math::max(acc,
                          alpaka::math::abs(acc, alpha_InLo),
                          alpaka::math::asin(acc, alpaka::math::min(acc, rt_InLo * k2Rinv1GeVf / 3.0f, kSinAlphaMax)));
    const float alphaOutAbsReg =
        alpaka::math::max(acc,
                          alpaka::math::abs(acc, alpha_OutLo),
                          alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * k2Rinv1GeVf / 3.0f, kSinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : alpaka::math::abs(acc, alphaInAbsReg * kDeltaZLum / z_InLo);
    const float dBetaOutLum = lOut < 11 ? 0.0f : alpaka::math::abs(acc, alphaOutAbsReg * kDeltaZLum / z_OutLo);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);
    const float sinDPhi = alpaka::math::sin(acc, dPhi);

    float dBetaROut = 0;
    if (isEC_lastLayer) {
      dBetaROut = (alpaka::math::sqrt(acc, highEdgeX_OutUp * highEdgeX_OutUp + highEdgeY_OutUp * highEdgeY_OutUp) -
                   alpaka::math::sqrt(acc, lowEdgeX_OutUp * lowEdgeX_OutUp + lowEdgeY_OutUp * lowEdgeY_OutUp)) *
                  sinDPhi / drt_tl_axis;
    }

    const float dBetaROut2 = dBetaROut * dBetaROut;

    float dBetaRes = 0.02f / alpaka::math::min(acc, sdOut_d, drt_InSeg);
    float dBetaCut2 =
        (dBetaRes * dBetaRes * 2.0f + dBetaMuls2 + dBetaLum2 + dBetaROut2 +
         0.25f *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)) *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)));

    dBeta = betaIn - betaOut;
    return dBeta * dBeta <= dBetaCut2;
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuintupletdBetaCutBBEE(TAcc const& acc,
                                                                ModulesConst modules,
                                                                MiniDoubletsConst mds,
                                                                SegmentsConst segments,
                                                                uint16_t innerInnerLowerModuleIndex,
                                                                uint16_t innerOuterLowerModuleIndex,
                                                                uint16_t outerInnerLowerModuleIndex,
                                                                uint16_t outerOuterLowerModuleIndex,
                                                                unsigned int innerSegmentIndex,
                                                                unsigned int outerSegmentIndex,
                                                                unsigned int firstMDIndex,
                                                                unsigned int secondMDIndex,
                                                                unsigned int thirdMDIndex,
                                                                unsigned int fourthMDIndex,
                                                                float& dBeta,
                                                                const float ptCut) {
    float rt_InLo = mds.anchorRt()[firstMDIndex];
    float rt_InOut = mds.anchorRt()[secondMDIndex];
    float rt_OutLo = mds.anchorRt()[thirdMDIndex];

    float z_InLo = mds.anchorZ()[firstMDIndex];
    float z_OutLo = mds.anchorZ()[thirdMDIndex];

    float rIn = alpaka::math::sqrt(acc, z_InLo * z_InLo + rt_InLo * rt_InLo);
    const float thetaMuls2 = (kMulsInGeV * kMulsInGeV) * (0.1f + 0.2f * (rt_OutLo - rt_InLo) / 50.f) * (rIn / rt_InLo);

    float midPointX = 0.5f * (mds.anchorX()[firstMDIndex] + mds.anchorX()[thirdMDIndex]);
    float midPointY = 0.5f * (mds.anchorY()[firstMDIndex] + mds.anchorY()[thirdMDIndex]);
    float diffX = mds.anchorX()[thirdMDIndex] - mds.anchorX()[firstMDIndex];
    float diffY = mds.anchorY()[thirdMDIndex] - mds.anchorY()[firstMDIndex];

    float dPhi = cms::alpakatools::deltaPhi(acc, midPointX, midPointY, diffX, diffY);

    float sdIn_alpha = __H2F(segments.dPhiChanges()[innerSegmentIndex]);
    float sdIn_alpha_min = __H2F(segments.dPhiChangeMins()[innerSegmentIndex]);
    float sdIn_alpha_max = __H2F(segments.dPhiChangeMaxs()[innerSegmentIndex]);
    float sdOut_alpha = sdIn_alpha;

    float sdOut_dPhiPos =
        cms::alpakatools::reducePhiRange(acc, mds.anchorPhi()[fourthMDIndex] - mds.anchorPhi()[thirdMDIndex]);

    float sdOut_dPhiChange = __H2F(segments.dPhiChanges()[outerSegmentIndex]);
    float sdOut_dPhiChange_min = __H2F(segments.dPhiChangeMins()[outerSegmentIndex]);
    float sdOut_dPhiChange_max = __H2F(segments.dPhiChangeMaxs()[outerSegmentIndex]);

    float sdOut_alphaOutRHmin = cms::alpakatools::reducePhiRange(acc, sdOut_dPhiChange_min - sdOut_dPhiPos);
    float sdOut_alphaOutRHmax = cms::alpakatools::reducePhiRange(acc, sdOut_dPhiChange_max - sdOut_dPhiPos);
    float sdOut_alphaOut = cms::alpakatools::reducePhiRange(acc, sdOut_dPhiChange - sdOut_dPhiPos);

    float tl_axis_x = mds.anchorX()[fourthMDIndex] - mds.anchorX()[firstMDIndex];
    float tl_axis_y = mds.anchorY()[fourthMDIndex] - mds.anchorY()[firstMDIndex];

    float betaIn =
        sdIn_alpha - cms::alpakatools::reducePhiRange(
                         acc, cms::alpakatools::phi(acc, tl_axis_x, tl_axis_y) - mds.anchorPhi()[firstMDIndex]);

    float betaInRHmin = betaIn;
    float betaInRHmax = betaIn;
    float betaOut =
        -sdOut_alphaOut + cms::alpakatools::reducePhiRange(
                              acc, cms::alpakatools::phi(acc, tl_axis_x, tl_axis_y) - mds.anchorPhi()[fourthMDIndex]);

    float betaOutRHmin = betaOut;
    float betaOutRHmax = betaOut;

    bool isEC_secondLayer = (modules.subdets()[innerOuterLowerModuleIndex] == Endcap) and
                            (modules.moduleType()[innerOuterLowerModuleIndex] == TwoS);

    if (isEC_secondLayer) {
      betaInRHmin = betaIn - sdIn_alpha_min + sdIn_alpha;
      betaInRHmax = betaIn - sdIn_alpha_max + sdIn_alpha;
    }

    betaOutRHmin = betaOut - sdOut_alphaOutRHmin + sdOut_alphaOut;
    betaOutRHmax = betaOut - sdOut_alphaOutRHmax + sdOut_alphaOut;

    float swapTemp;
    if (alpaka::math::abs(acc, betaOutRHmin) > alpaka::math::abs(acc, betaOutRHmax)) {
      swapTemp = betaOutRHmin;
      betaOutRHmin = betaOutRHmax;
      betaOutRHmax = swapTemp;
    }

    if (alpaka::math::abs(acc, betaInRHmin) > alpaka::math::abs(acc, betaInRHmax)) {
      swapTemp = betaInRHmin;
      betaInRHmin = betaInRHmax;
      betaInRHmax = swapTemp;
    }

    float sdIn_dr = alpaka::math::sqrt(acc,
                                       (mds.anchorX()[secondMDIndex] - mds.anchorX()[firstMDIndex]) *
                                               (mds.anchorX()[secondMDIndex] - mds.anchorX()[firstMDIndex]) +
                                           (mds.anchorY()[secondMDIndex] - mds.anchorY()[firstMDIndex]) *
                                               (mds.anchorY()[secondMDIndex] - mds.anchorY()[firstMDIndex]));
    float sdIn_d = rt_InOut - rt_InLo;

    float dr = alpaka::math::sqrt(acc, tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);

    float betaAv = 0.5f * (betaIn + betaOut);
    float pt_beta = dr * k2Rinv1GeVf / alpaka::math::sin(acc, betaAv);

    float lIn = 5;
    float lOut = 11;

    float sdOut_dr = alpaka::math::sqrt(acc,
                                        (mds.anchorX()[fourthMDIndex] - mds.anchorX()[thirdMDIndex]) *
                                                (mds.anchorX()[fourthMDIndex] - mds.anchorX()[thirdMDIndex]) +
                                            (mds.anchorY()[fourthMDIndex] - mds.anchorY()[thirdMDIndex]) *
                                                (mds.anchorY()[fourthMDIndex] - mds.anchorY()[thirdMDIndex]));
    float sdOut_d = mds.anchorRt()[fourthMDIndex] - mds.anchorRt()[thirdMDIndex];

    runDeltaBetaIterations(acc, betaIn, betaOut, pt_beta, sdIn_dr, sdOut_dr, dr, lIn);

    const float betaInMMSF = (alpaka::math::abs(acc, betaInRHmin + betaInRHmax) > 0)
                                 ? (2.f * betaIn / alpaka::math::abs(acc, betaInRHmin + betaInRHmax))
                                 : 0.;  //mean value of min,max is the old betaIn
    const float betaOutMMSF = (alpaka::math::abs(acc, betaOutRHmin + betaOutRHmax) > 0)
                                  ? (2.f * betaOut / alpaka::math::abs(acc, betaOutRHmin + betaOutRHmax))
                                  : 0.;
    betaInRHmin *= betaInMMSF;
    betaInRHmax *= betaInMMSF;
    betaOutRHmin *= betaOutMMSF;
    betaOutRHmax *= betaOutMMSF;

    float min_ptBeta_maxPtBeta = alpaka::math::min(
        acc, alpaka::math::abs(acc, pt_beta), kPt_betaMax);  //need to confirm the range-out value of 7 GeV
    const float dBetaMuls2 = thetaMuls2 * 16.f / (min_ptBeta_maxPtBeta * min_ptBeta_maxPtBeta);

    const float alphaInAbsReg =
        alpaka::math::max(acc,
                          alpaka::math::abs(acc, sdIn_alpha),
                          alpaka::math::asin(acc, alpaka::math::min(acc, rt_InLo * k2Rinv1GeVf / 3.0f, kSinAlphaMax)));
    const float alphaOutAbsReg =
        alpaka::math::max(acc,
                          alpaka::math::abs(acc, sdOut_alpha),
                          alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * k2Rinv1GeVf / 3.0f, kSinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : alpaka::math::abs(acc, alphaInAbsReg * kDeltaZLum / z_InLo);
    const float dBetaOutLum = lOut < 11 ? 0.0f : alpaka::math::abs(acc, alphaOutAbsReg * kDeltaZLum / z_OutLo);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);
    const float sinDPhi = alpaka::math::sin(acc, dPhi);

    const float dBetaRIn2 = 0;  // TODO-RH
    float dBetaROut = 0;
    if (modules.moduleType()[outerOuterLowerModuleIndex] == TwoS) {
      // outer MD strip edges: anchor xy +- the module's strip half-vector
      const float highEdgeX_OutUp = mds.anchorX()[fourthMDIndex] + modules.edgeDx()[outerOuterLowerModuleIndex];
      const float highEdgeY_OutUp = mds.anchorY()[fourthMDIndex] + modules.edgeDy()[outerOuterLowerModuleIndex];
      const float lowEdgeX_OutUp = mds.anchorX()[fourthMDIndex] - modules.edgeDx()[outerOuterLowerModuleIndex];
      const float lowEdgeY_OutUp = mds.anchorY()[fourthMDIndex] - modules.edgeDy()[outerOuterLowerModuleIndex];
      dBetaROut = (alpaka::math::sqrt(acc, highEdgeX_OutUp * highEdgeX_OutUp + highEdgeY_OutUp * highEdgeY_OutUp) -
                   alpaka::math::sqrt(acc, lowEdgeX_OutUp * lowEdgeX_OutUp + lowEdgeY_OutUp * lowEdgeY_OutUp)) *
                  sinDPhi / dr;
    }

    const float dBetaROut2 = dBetaROut * dBetaROut;

    float dBetaRes = 0.02f / alpaka::math::min(acc, sdOut_d, sdIn_d);
    float dBetaCut2 =
        (dBetaRes * dBetaRes * 2.0f + dBetaMuls2 + dBetaLum2 + dBetaRIn2 + dBetaROut2 +
         0.25f *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)) *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)));
    dBeta = betaIn - betaOut;
    return dBeta * dBeta <= dBetaCut2;
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuintupletdBetaCutEEEE(TAcc const& acc,
                                                                ModulesConst modules,
                                                                MiniDoubletsConst mds,
                                                                SegmentsConst segments,
                                                                uint16_t innerInnerLowerModuleIndex,
                                                                uint16_t innerOuterLowerModuleIndex,
                                                                uint16_t outerInnerLowerModuleIndex,
                                                                uint16_t outerOuterLowerModuleIndex,
                                                                unsigned int innerSegmentIndex,
                                                                unsigned int outerSegmentIndex,
                                                                unsigned int firstMDIndex,
                                                                unsigned int secondMDIndex,
                                                                unsigned int thirdMDIndex,
                                                                unsigned int fourthMDIndex,
                                                                float& dBeta,
                                                                const float ptCut) {
    float rt_InLo = mds.anchorRt()[firstMDIndex];
    float rt_InOut = mds.anchorRt()[secondMDIndex];
    float rt_OutLo = mds.anchorRt()[thirdMDIndex];

    float z_InLo = mds.anchorZ()[firstMDIndex];
    float z_OutLo = mds.anchorZ()[thirdMDIndex];

    float thetaMuls2 = (kMulsInGeV * kMulsInGeV) * (0.1f + 0.2f * (rt_OutLo - rt_InLo) / 50.f);
    float sdIn_alpha = __H2F(segments.dPhiChanges()[innerSegmentIndex]);
    float sdOut_alpha = sdIn_alpha;  //weird
    float sdOut_dPhiPos =
        cms::alpakatools::reducePhiRange(acc, mds.anchorPhi()[fourthMDIndex] - mds.anchorPhi()[thirdMDIndex]);

    float sdOut_dPhiChange = __H2F(segments.dPhiChanges()[outerSegmentIndex]);
    float sdOut_dPhiChange_min = __H2F(segments.dPhiChangeMins()[outerSegmentIndex]);
    float sdOut_dPhiChange_max = __H2F(segments.dPhiChangeMaxs()[outerSegmentIndex]);

    float sdOut_alphaOutRHmin = cms::alpakatools::reducePhiRange(acc, sdOut_dPhiChange_min - sdOut_dPhiPos);
    float sdOut_alphaOutRHmax = cms::alpakatools::reducePhiRange(acc, sdOut_dPhiChange_max - sdOut_dPhiPos);
    float sdOut_alphaOut = cms::alpakatools::reducePhiRange(acc, sdOut_dPhiChange - sdOut_dPhiPos);

    float tl_axis_x = mds.anchorX()[fourthMDIndex] - mds.anchorX()[firstMDIndex];
    float tl_axis_y = mds.anchorY()[fourthMDIndex] - mds.anchorY()[firstMDIndex];

    float betaIn =
        sdIn_alpha - cms::alpakatools::reducePhiRange(
                         acc, cms::alpakatools::phi(acc, tl_axis_x, tl_axis_y) - mds.anchorPhi()[firstMDIndex]);

    float sdIn_alphaRHmin = __H2F(segments.dPhiChangeMins()[innerSegmentIndex]);
    float sdIn_alphaRHmax = __H2F(segments.dPhiChangeMaxs()[innerSegmentIndex]);
    float betaInRHmin = betaIn + sdIn_alphaRHmin - sdIn_alpha;
    float betaInRHmax = betaIn + sdIn_alphaRHmax - sdIn_alpha;

    float betaOut =
        -sdOut_alphaOut + cms::alpakatools::reducePhiRange(
                              acc, cms::alpakatools::phi(acc, tl_axis_x, tl_axis_y) - mds.anchorPhi()[fourthMDIndex]);

    float betaOutRHmin = betaOut - sdOut_alphaOutRHmin + sdOut_alphaOut;
    float betaOutRHmax = betaOut - sdOut_alphaOutRHmax + sdOut_alphaOut;

    float swapTemp;
    if (alpaka::math::abs(acc, betaOutRHmin) > alpaka::math::abs(acc, betaOutRHmax)) {
      swapTemp = betaOutRHmin;
      betaOutRHmin = betaOutRHmax;
      betaOutRHmax = swapTemp;
    }

    if (alpaka::math::abs(acc, betaInRHmin) > alpaka::math::abs(acc, betaInRHmax)) {
      swapTemp = betaInRHmin;
      betaInRHmin = betaInRHmax;
      betaInRHmax = swapTemp;
    }
    float sdIn_dr = alpaka::math::sqrt(acc,
                                       (mds.anchorX()[secondMDIndex] - mds.anchorX()[firstMDIndex]) *
                                               (mds.anchorX()[secondMDIndex] - mds.anchorX()[firstMDIndex]) +
                                           (mds.anchorY()[secondMDIndex] - mds.anchorY()[firstMDIndex]) *
                                               (mds.anchorY()[secondMDIndex] - mds.anchorY()[firstMDIndex]));
    float sdIn_d = rt_InOut - rt_InLo;

    float dr = alpaka::math::sqrt(acc, tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);

    float betaAv = 0.5f * (betaIn + betaOut);
    float pt_beta = dr * k2Rinv1GeVf / alpaka::math::sin(acc, betaAv);

    int lIn = 11;   //endcap
    int lOut = 13;  //endcap

    float sdOut_dr = alpaka::math::sqrt(acc,
                                        (mds.anchorX()[fourthMDIndex] - mds.anchorX()[thirdMDIndex]) *
                                                (mds.anchorX()[fourthMDIndex] - mds.anchorX()[thirdMDIndex]) +
                                            (mds.anchorY()[fourthMDIndex] - mds.anchorY()[thirdMDIndex]) *
                                                (mds.anchorY()[fourthMDIndex] - mds.anchorY()[thirdMDIndex]));
    float sdOut_d = mds.anchorRt()[fourthMDIndex] - mds.anchorRt()[thirdMDIndex];

    runDeltaBetaIterations(acc, betaIn, betaOut, pt_beta, sdIn_dr, sdOut_dr, dr, lIn);

    const float betaInMMSF = (alpaka::math::abs(acc, betaInRHmin + betaInRHmax) > 0)
                                 ? (2.f * betaIn / alpaka::math::abs(acc, betaInRHmin + betaInRHmax))
                                 : 0.;  //mean value of min,max is the old betaIn
    const float betaOutMMSF = (alpaka::math::abs(acc, betaOutRHmin + betaOutRHmax) > 0)
                                  ? (2.f * betaOut / alpaka::math::abs(acc, betaOutRHmin + betaOutRHmax))
                                  : 0.;
    betaInRHmin *= betaInMMSF;
    betaInRHmax *= betaInMMSF;
    betaOutRHmin *= betaOutMMSF;
    betaOutRHmax *= betaOutMMSF;

    float min_ptBeta_maxPtBeta = alpaka::math::min(
        acc, alpaka::math::abs(acc, pt_beta), kPt_betaMax);  //need to confirm the range-out value of 7 GeV
    const float dBetaMuls2 = thetaMuls2 * 16.f / (min_ptBeta_maxPtBeta * min_ptBeta_maxPtBeta);

    const float alphaInAbsReg =
        alpaka::math::max(acc,
                          alpaka::math::abs(acc, sdIn_alpha),
                          alpaka::math::asin(acc, alpaka::math::min(acc, rt_InLo * k2Rinv1GeVf / 3.0f, kSinAlphaMax)));
    const float alphaOutAbsReg =
        alpaka::math::max(acc,
                          alpaka::math::abs(acc, sdOut_alpha),
                          alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * k2Rinv1GeVf / 3.0f, kSinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : alpaka::math::abs(acc, alphaInAbsReg * kDeltaZLum / z_InLo);
    const float dBetaOutLum = lOut < 11 ? 0.0f : alpaka::math::abs(acc, alphaOutAbsReg * kDeltaZLum / z_OutLo);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);

    float dBetaRes = 0.02f / alpaka::math::min(acc, sdOut_d, sdIn_d);
    float dBetaCut2 =
        (dBetaRes * dBetaRes * 2.0f + dBetaMuls2 + dBetaLum2 +
         0.25f *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)) *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)));
    dBeta = betaIn - betaOut;
    return dBeta * dBeta <= dBetaCut2;
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuintupletdBetaAlgoSelector(TAcc const& acc,
                                                                     ModulesConst modules,
                                                                     MiniDoubletsConst mds,
                                                                     SegmentsConst segments,
                                                                     uint16_t innerInnerLowerModuleIndex,
                                                                     uint16_t innerOuterLowerModuleIndex,
                                                                     uint16_t outerInnerLowerModuleIndex,
                                                                     uint16_t outerOuterLowerModuleIndex,
                                                                     unsigned int innerSegmentIndex,
                                                                     unsigned int outerSegmentIndex,
                                                                     unsigned int firstMDIndex,
                                                                     unsigned int secondMDIndex,
                                                                     unsigned int thirdMDIndex,
                                                                     unsigned int fourthMDIndex,
                                                                     float& dBeta,
                                                                     const float ptCut) {
    short innerInnerLowerModuleSubdet = modules.subdets()[innerInnerLowerModuleIndex];
    short innerOuterLowerModuleSubdet = modules.subdets()[innerOuterLowerModuleIndex];
    short outerInnerLowerModuleSubdet = modules.subdets()[outerInnerLowerModuleIndex];
    short outerOuterLowerModuleSubdet = modules.subdets()[outerOuterLowerModuleIndex];

    if (innerInnerLowerModuleSubdet == Barrel and innerOuterLowerModuleSubdet == Barrel and
        outerInnerLowerModuleSubdet == Barrel and outerOuterLowerModuleSubdet == Barrel) {
      return runQuintupletdBetaCutBBBB(acc,
                                       modules,
                                       mds,
                                       segments,
                                       innerInnerLowerModuleIndex,
                                       innerOuterLowerModuleIndex,
                                       outerInnerLowerModuleIndex,
                                       outerOuterLowerModuleIndex,
                                       innerSegmentIndex,
                                       outerSegmentIndex,
                                       firstMDIndex,
                                       secondMDIndex,
                                       thirdMDIndex,
                                       fourthMDIndex,
                                       dBeta,
                                       ptCut);
    } else if (innerInnerLowerModuleSubdet == Barrel and innerOuterLowerModuleSubdet == Barrel and
               outerInnerLowerModuleSubdet == Endcap and outerOuterLowerModuleSubdet == Endcap) {
      return runQuintupletdBetaCutBBEE(acc,
                                       modules,
                                       mds,
                                       segments,
                                       innerInnerLowerModuleIndex,
                                       innerOuterLowerModuleIndex,
                                       outerInnerLowerModuleIndex,
                                       outerOuterLowerModuleIndex,
                                       innerSegmentIndex,
                                       outerSegmentIndex,
                                       firstMDIndex,
                                       secondMDIndex,
                                       thirdMDIndex,
                                       fourthMDIndex,
                                       dBeta,
                                       ptCut);
    } else if (innerInnerLowerModuleSubdet == Barrel and innerOuterLowerModuleSubdet == Barrel and
               outerInnerLowerModuleSubdet == Barrel and outerOuterLowerModuleSubdet == Endcap) {
      return runQuintupletdBetaCutBBBB(acc,
                                       modules,
                                       mds,
                                       segments,
                                       innerInnerLowerModuleIndex,
                                       innerOuterLowerModuleIndex,
                                       outerInnerLowerModuleIndex,
                                       outerOuterLowerModuleIndex,
                                       innerSegmentIndex,
                                       outerSegmentIndex,
                                       firstMDIndex,
                                       secondMDIndex,
                                       thirdMDIndex,
                                       fourthMDIndex,
                                       dBeta,
                                       ptCut);
    } else if (innerInnerLowerModuleSubdet == Barrel and innerOuterLowerModuleSubdet == Endcap and
               outerInnerLowerModuleSubdet == Endcap and outerOuterLowerModuleSubdet == Endcap) {
      return runQuintupletdBetaCutBBEE(acc,
                                       modules,
                                       mds,
                                       segments,
                                       innerInnerLowerModuleIndex,
                                       innerOuterLowerModuleIndex,
                                       outerInnerLowerModuleIndex,
                                       outerOuterLowerModuleIndex,
                                       innerSegmentIndex,
                                       outerSegmentIndex,
                                       firstMDIndex,
                                       secondMDIndex,
                                       thirdMDIndex,
                                       fourthMDIndex,
                                       dBeta,
                                       ptCut);
    } else if (innerInnerLowerModuleSubdet == Endcap and innerOuterLowerModuleSubdet == Endcap and
               outerInnerLowerModuleSubdet == Endcap and outerOuterLowerModuleSubdet == Endcap) {
      return runQuintupletdBetaCutEEEE(acc,
                                       modules,
                                       mds,
                                       segments,
                                       innerInnerLowerModuleIndex,
                                       innerOuterLowerModuleIndex,
                                       outerInnerLowerModuleIndex,
                                       outerOuterLowerModuleIndex,
                                       innerSegmentIndex,
                                       outerSegmentIndex,
                                       firstMDIndex,
                                       secondMDIndex,
                                       thirdMDIndex,
                                       fourthMDIndex,
                                       dBeta,
                                       ptCut);
    }

    return false;
  }

  // The cuts of a T5 that depend only on the inner T3 and the outer T3's first segment (outer LS1, thirdSegmentIndex):
  // the first dBeta cut. The T5 builders apply them once per (inner T3, outer LS1).
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passInnerT3OuterLS1Cuts(TAcc const& acc,
                                                              ModulesConst modules,
                                                              MiniDoubletsConst mds,
                                                              SegmentsConst segments,
                                                              TripletsConst triplets,
                                                              uint16_t lowerModuleIndex1,
                                                              unsigned int innerTripletIndex,
                                                              unsigned int thirdSegmentIndex,
                                                              float& dBeta1,
                                                              const float ptCut) {
    const uint16_t lowerModuleIndex2 = triplets.lowerModuleIndices()[innerTripletIndex][1];
    const uint16_t lowerModuleIndex3 = triplets.lowerModuleIndices()[innerTripletIndex][2];
    const uint16_t lowerModuleIndex4 = segments.outerLowerModuleIndices()[thirdSegmentIndex];
    const unsigned int firstSegmentIndex = triplets.segmentIndices()[innerTripletIndex][0];
    const unsigned int secondSegmentIndex = triplets.segmentIndices()[innerTripletIndex][1];
    const unsigned int firstMDIndex = segments.mdIndices()[firstSegmentIndex][0];
    const unsigned int secondMDIndex = segments.mdIndices()[secondSegmentIndex][0];
    const unsigned int thirdMDIndex = segments.mdIndices()[secondSegmentIndex][1];
    const unsigned int fourthMDIndex = segments.mdIndices()[thirdSegmentIndex][1];
    return runQuintupletdBetaAlgoSelector(acc,
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
                                          dBeta1,
                                          ptCut);
  }

  // The cuts of a T5 that depend only on the inner T3's first segment and the outer T3's second segment (outer LS2):
  // the second dBeta cut.
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passInnerT3OuterLS2Cuts(TAcc const& acc,
                                                              ModulesConst modules,
                                                              MiniDoubletsConst mds,
                                                              SegmentsConst segments,
                                                              TripletsConst triplets,
                                                              uint16_t lowerModuleIndex1,
                                                              unsigned int innerTripletIndex,
                                                              unsigned int outerTripletIndex,
                                                              float& dBeta2,
                                                              const float ptCut) {
    const uint16_t lowerModuleIndex2 = triplets.lowerModuleIndices()[innerTripletIndex][1];
    const uint16_t lowerModuleIndex4 = triplets.lowerModuleIndices()[outerTripletIndex][1];
    const uint16_t lowerModuleIndex5 = triplets.lowerModuleIndices()[outerTripletIndex][2];
    const unsigned int firstSegmentIndex = triplets.segmentIndices()[innerTripletIndex][0];
    const unsigned int secondSegmentIndex = triplets.segmentIndices()[innerTripletIndex][1];
    const unsigned int thirdSegmentIndex = triplets.segmentIndices()[outerTripletIndex][0];
    const unsigned int fourthSegmentIndex = triplets.segmentIndices()[outerTripletIndex][1];
    const unsigned int firstMDIndex = segments.mdIndices()[firstSegmentIndex][0];
    const unsigned int secondMDIndex = segments.mdIndices()[secondSegmentIndex][0];
    const unsigned int fourthMDIndex = segments.mdIndices()[thirdSegmentIndex][1];
    const unsigned int fifthMDIndex = segments.mdIndices()[fourthSegmentIndex][1];
    return runQuintupletdBetaAlgoSelector(acc,
                                          modules,
                                          mds,
                                          segments,
                                          lowerModuleIndex1,
                                          lowerModuleIndex2,
                                          lowerModuleIndex4,
                                          lowerModuleIndex5,
                                          firstSegmentIndex,
                                          fourthSegmentIndex,
                                          firstMDIndex,
                                          secondMDIndex,
                                          fourthMDIndex,
                                          fifthMDIndex,
                                          dBeta2,
                                          ptCut);
  }

  // passInnerT3OuterLS2Cuts depends only on the inner T3's first segment and the outer LS2. outerLS2PassCache[outer LS2]
  // keeps the decision for the last first-inner segment it was tested with, as firstSegment << 1 | pass (0xFFFFFFFF =
  // empty), so a repeated (first-inner segment, outer LS2) pair reuses it. Threads of different blocks read and write
  // an entry without atomics on purpose: each 32-bit word is a self-consistent (segment, decision) pair written by one
  // aligned store, and a reader uses it only if the segment matches, so a value from another writer is only a miss.
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passInnerT3OuterLS2CutsCached(TAcc const& acc,
                                                                    ModulesConst modules,
                                                                    MiniDoubletsConst mds,
                                                                    SegmentsConst segments,
                                                                    TripletsConst triplets,
                                                                    uint16_t lowerModuleIndex1,
                                                                    unsigned int innerTripletIndex,
                                                                    unsigned int outerTripletIndex,
                                                                    const float ptCut,
                                                                    unsigned int* __restrict__ outerLS2PassCache) {
    const unsigned int tag = triplets.segmentIndices()[innerTripletIndex][0] << 1;
    const unsigned int fourthSegmentIndex = triplets.segmentIndices()[outerTripletIndex][1];
    const unsigned int cached = outerLS2PassCache[fourthSegmentIndex];
    if ((cached & ~1u) == tag)
      return cached & 1u;
    float dBeta2;
    const bool pass = passInnerT3OuterLS2Cuts(
        acc, modules, mds, segments, triplets, lowerModuleIndex1, innerTripletIndex, outerTripletIndex, dBeta2, ptCut);
    outerLS2PassCache[fourthSegmentIndex] = tag | (pass ? 1u : 0u);
    return pass;
  }

  // Radius of the circle through the anchor hits of MDs 2, 3, 4 (inner triplet and the outer triplet's first segment).
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computeT5BridgeRadius(TAcc const& acc,
                                                             MiniDoubletsConst mds,
                                                             SegmentsConst segments,
                                                             TripletsConst triplets,
                                                             unsigned int innerTripletIndex,
                                                             unsigned int thirdSegmentIndex) {
    const unsigned int secondSegmentIndex = triplets.segmentIndices()[innerTripletIndex][1];
    const unsigned int secondMDIndex = segments.mdIndices()[secondSegmentIndex][0];
    const unsigned int thirdMDIndex = segments.mdIndices()[secondSegmentIndex][1];
    const unsigned int fourthMDIndex = segments.mdIndices()[thirdSegmentIndex][1];
    return std::get<0>(computeRadiusFromThreeAnchorHits(acc,
                                                        mds.anchorX()[secondMDIndex],
                                                        mds.anchorY()[secondMDIndex],
                                                        mds.anchorX()[thirdMDIndex],
                                                        mds.anchorY()[thirdMDIndex],
                                                        mds.anchorX()[fourthMDIndex],
                                                        mds.anchorY()[fourthMDIndex]));
  }

  // T5/T4 DNN inputs over the N MDs: MD-direction log-likelihood (mean, max; module frame, circle through anchors
  // iA,iB,iC), local density (T3s leaving MD iMid and MD 0, MDs in the first module) and dcaXY of that circle.
  // Not inlined on ROCm: inlining it into the counting kernels crashes the gfx90a register allocator (ROCm 7.2).
#if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#define LST_DNN_FEATURES_INLINE [[gnu::noinline]]
#else
#define LST_DNN_FEATURES_INLINE ALPAKA_FN_INLINE
#endif
  template <int N, int iA, int iB, int iC, int iMid, alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC LST_DNN_FEATURES_INLINE
      dnn::t5dnn::ExtraFeatures computeDnnFeatures(TAcc const& acc,
                                                   ModulesConst modules,
                                                   MiniDoubletsConst mds,
                                                   MiniDoubletsOccupancyConst mdOccupancy,
                                                   TripletCountsConst tripletCountsByMD,
                                                   const uint16_t (&lm)[N],
                                                   const unsigned int (&md)[N]) {
    float ax[N], ay[N], az[N];
    for (int i = 0; i < N; ++i) {
      ax[i] = mds.anchorX()[md[i]];
      ay[i] = mds.anchorY()[md[i]];
      az[i] = mds.anchorZ()[md[i]];
    }
    const auto circle = computeRadiusFromThreeAnchorHits(acc, ax[iA], ay[iA], ax[iB], ay[iB], ax[iC], ay[iC]);
    const float cr = std::get<0>(circle), cx = std::get<1>(circle), cy = std::get<2>(circle);
    const float chx = ax[iC] - ax[iA], chy = ay[iC] - ay[iA];
    const float chord = alpaka::math::sqrt(acc, chx * chx + chy * chy);
    // Same radius as cr, measured from the first anchor: numerically safer than cr for nearly straight tracks.
    const float radiusAtFirstAnchor =
        alpaka::math::sqrt(acc, (ax[iA] - cx) * (ax[iA] - cx) + (ay[iA] - cy) * (ay[iA] - cy));
    float arc = chord;
    if (edm::isFinite(radiusAtFirstAnchor) && radiusAtFirstAnchor > 0.f)
      arc = 2.f * radiusAtFirstAnchor *
            alpaka::math::asin(acc, alpaka::math::min(acc, chord / (2.f * radiusAtFirstAnchor), 1.f));
    const float cotTheta = (az[iC] - az[iA]) / arc;
    float sumW = 0.f, maxW = 0.f, maxPull = 0.f;
    for (int i = 0; i < N; ++i) {
      const uint16_t lowerModuleIndex = lm[i];
      float tx = cy - ay[i], ty = ax[i] - cx;
      const float tn = alpaka::math::sqrt(acc, tx * tx + ty * ty);
      tx /= tn;
      ty /= tn;
      if (!edm::isFinite(tx) || !edm::isFinite(ty)) {
        tx = chx / chord;
        ty = chy / chord;
      }
      if (tx * chx + ty * chy < 0.f) {
        tx = -tx;
        ty = -ty;
      }
      // t = unit tangent of the circle at the MD anchor (plus cotTheta along z); in the module frame,
      // u = direction across the strips in the sensor plane, n = sensor normal (tilted barrel via drdz, endcap = z).
      float ux, uy, nx, ny, nz;
      if (modules.subdets()[lowerModuleIndex] == Barrel) {
        const float cphi = alpaka::math::cos(acc, modules.phi()[lowerModuleIndex]);
        const float sphi = alpaka::math::sin(acc, modules.phi()[lowerModuleIndex]);
        ux = -sphi;
        uy = cphi;
        nx = cphi;
        ny = sphi;
        nz = ((az[i] > 0.f) - (az[i] < 0.f)) * modules.drdzs()[lowerModuleIndex];
      } else {
        const float slope = modules.dxdys()[lowerModuleIndex];
        ux = 0.f;
        uy = 1.f;
        if (edm::isFinite(slope)) {
          ux = 1.f / alpaka::math::sqrt(acc, 1.f + slope * slope);
          uy = slope * ux;
        }
        nx = 0.f;
        ny = 0.f;
        nz = 1.f;
      }
      const float dx = mds.outerX()[md[i]] - ax[i];
      const float dy = mds.outerY()[md[i]] - ay[i];
      const float dz = mds.outerZ()[md[i]] - az[i];
      const float vu = tx * ux + ty * uy;
      const float vn = tx * nx + ty * ny + cotTheta * nz;
      const float residual = (dx * ux + dy * uy) - vu * (dx * nx + dy * ny + dz * nz) / vn;
      const float width = (modules.moduleType()[lowerModuleIndex] == PS) ? kWidthPS : kWidth2S;
      const float pull = alpaka::math::abs(acc, residual) / (width * 0.40824829f);
      const float mdDirW = -alpaka::math::log(acc, alpaka::math::max(acc, 1.f - pull * 0.40824829f, 0.05f));
      sumW += mdDirW;
      maxW = alpaka::math::max(acc, maxW, mdDirW);
      maxPull = alpaka::math::max(acc, maxPull, pull);
    }
    dnn::t5dnn::ExtraFeatures feat;
    feat.mdDirMeanW = sumW / N;
    feat.mdDirMaxW = maxW;
    feat.nT3OutMid = tripletCountsByMD.n()[md[iMid]];
    feat.nT3OutFirst = tripletCountsByMD.n()[md[0]];
    feat.nMDFirstMod = mdOccupancy.nMDs()[lm[0]];
    feat.dcaXY = alpaka::math::abs(acc, alpaka::math::sqrt(acc, cx * cx + cy * cy) - cr);
    feat.mdDirMaxPull = maxPull;
    return feat;
  }

  // The cuts of the T5 algorithm after the outer LS1 and LS2 cuts (bridgeRadius is computed per (inner T3, outer LS1));
  // computeQuintupletExtras adds the embedding and fits of a selected T5.
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuintupletSelectionOuter(TAcc const& acc,
                                                                  ModulesConst modules,
                                                                  MiniDoubletsConst mds,
                                                                  SegmentsConst segments,
                                                                  TripletsConst triplets,
                                                                  MiniDoubletsOccupancyConst mdOccupancy,
                                                                  TripletCountsConst tripletCountsByMD,
                                                                  uint16_t lowerModuleIndex1,
                                                                  unsigned int innerTripletIndex,
                                                                  unsigned int outerTripletIndex,
                                                                  float bridgeRadius,
                                                                  float& innerRadius,
                                                                  float& outerRadius,
                                                                  float& rzChiSquared,
                                                                  float& dnnScore) {
    const uint16_t lowerModuleIndex2 = triplets.lowerModuleIndices()[innerTripletIndex][1];
    const uint16_t lowerModuleIndex3 = triplets.lowerModuleIndices()[innerTripletIndex][2];
    const uint16_t lowerModuleIndex4 = triplets.lowerModuleIndices()[outerTripletIndex][1];
    const uint16_t lowerModuleIndex5 = triplets.lowerModuleIndices()[outerTripletIndex][2];

    unsigned int firstSegmentIndex = triplets.segmentIndices()[innerTripletIndex][0];
    unsigned int secondSegmentIndex = triplets.segmentIndices()[innerTripletIndex][1];
    unsigned int thirdSegmentIndex = triplets.segmentIndices()[outerTripletIndex][0];
    unsigned int fourthSegmentIndex = triplets.segmentIndices()[outerTripletIndex][1];

    unsigned int firstMDIndex = segments.mdIndices()[firstSegmentIndex][0];
    unsigned int secondMDIndex = segments.mdIndices()[secondSegmentIndex][0];
    unsigned int thirdMDIndex = segments.mdIndices()[secondSegmentIndex][1];
    unsigned int fourthMDIndex = segments.mdIndices()[thirdSegmentIndex][1];
    unsigned int fifthMDIndex = segments.mdIndices()[fourthSegmentIndex][1];

    outerRadius = triplets.radius()[outerTripletIndex];
    innerRadius = triplets.radius()[innerTripletIndex];

    float g = triplets.centerX()[innerTripletIndex];
    float f = triplets.centerY()[innerTripletIndex];

    float inner_pt = 2 * k2Rinv1GeVf * innerRadius;

    if (not passT5RZConstraint(acc,
                               modules,
                               mds,
                               firstMDIndex,
                               secondMDIndex,
                               thirdMDIndex,
                               fourthMDIndex,
                               fifthMDIndex,
                               lowerModuleIndex1,
                               lowerModuleIndex2,
                               lowerModuleIndex3,
                               lowerModuleIndex4,
                               lowerModuleIndex5,
                               rzChiSquared,
                               inner_pt,
                               innerRadius,
                               g,
                               f))
      return false;

    const uint16_t t5Lm[Params_T5::kBaseLayers] = {
        lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5};
    const unsigned int t5Md[Params_T5::kBaseLayers] = {
        firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, fifthMDIndex};
    const auto t5Feat = computeDnnFeatures<Params_T5::kBaseLayers, 0, 2, 4, 2>(
        acc, modules, mds, mdOccupancy, tripletCountsByMD, t5Lm, t5Md);
    float dnnOutput[dnn::t5dnn::kOutputFeatures];
    const bool inference = lst::t5dnn::runInference(acc,
                                                    mds,
                                                    firstMDIndex,
                                                    secondMDIndex,
                                                    thirdMDIndex,
                                                    fourthMDIndex,
                                                    fifthMDIndex,
                                                    innerRadius,
                                                    outerRadius,
                                                    bridgeRadius,
                                                    triplets.fakeScore()[innerTripletIndex],
                                                    triplets.promptScore()[innerTripletIndex],
                                                    triplets.displacedScore()[innerTripletIndex],
                                                    triplets.fakeScore()[outerTripletIndex],
                                                    triplets.promptScore()[outerTripletIndex],
                                                    triplets.displacedScore()[outerTripletIndex],
                                                    t5Feat,
                                                    dnnOutput);
    dnnScore = 1.f - dnnOutput[0];
    if (!inference)  // T5-building cut
      return false;
    return true;
  }

  // Every cut of the T5 algorithm for one (inner, outer) triplet pair.
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuintupletSelection(TAcc const& acc,
                                                             ModulesConst modules,
                                                             MiniDoubletsConst mds,
                                                             SegmentsConst segments,
                                                             TripletsConst triplets,
                                                             MiniDoubletsOccupancyConst mdOccupancy,
                                                             TripletCountsConst tripletCountsByMD,
                                                             uint16_t lowerModuleIndex1,
                                                             unsigned int innerTripletIndex,
                                                             unsigned int outerTripletIndex,
                                                             float& innerRadius,
                                                             float& outerRadius,
                                                             float& bridgeRadius,
                                                             float& rzChiSquared,
                                                             float& dBeta1,
                                                             float& dBeta2,
                                                             float& dnnScore,
                                                             const float ptCut) {
    const unsigned int thirdSegmentIndex = triplets.segmentIndices()[outerTripletIndex][0];
    bridgeRadius = computeT5BridgeRadius(acc, mds, segments, triplets, innerTripletIndex, thirdSegmentIndex);
    if (not passInnerT3OuterLS1Cuts(acc,
                                    modules,
                                    mds,
                                    segments,
                                    triplets,
                                    lowerModuleIndex1,
                                    innerTripletIndex,
                                    thirdSegmentIndex,
                                    dBeta1,
                                    ptCut))
      return false;
    if (not passInnerT3OuterLS2Cuts(acc,
                                    modules,
                                    mds,
                                    segments,
                                    triplets,
                                    lowerModuleIndex1,
                                    innerTripletIndex,
                                    outerTripletIndex,
                                    dBeta2,
                                    ptCut))
      return false;
    return runQuintupletSelectionOuter(acc,
                                       modules,
                                       mds,
                                       segments,
                                       triplets,
                                       mdOccupancy,
                                       tripletCountsByMD,
                                       lowerModuleIndex1,
                                       innerTripletIndex,
                                       outerTripletIndex,
                                       bridgeRadius,
                                       innerRadius,
                                       outerRadius,
                                       rzChiSquared,
                                       dnnScore);
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void computeQuintupletExtras(TAcc const& acc,
                                                              ModulesConst modules,
                                                              MiniDoubletsConst mds,
                                                              SegmentsConst segments,
                                                              TripletsConst triplets,
                                                              uint16_t lowerModuleIndex1,
                                                              uint16_t lowerModuleIndex2,
                                                              uint16_t lowerModuleIndex3,
                                                              uint16_t lowerModuleIndex4,
                                                              uint16_t lowerModuleIndex5,
                                                              unsigned int innerTripletIndex,
                                                              unsigned int outerTripletIndex,
                                                              float innerRadius,
                                                              float outerRadius,
                                                              float bridgeRadius,
                                                              float& regressionCenterX,
                                                              float& regressionCenterY,
                                                              float& regressionRadius,
                                                              float& chiSquared,
                                                              float (&t5Embed)[Params_T5::kEmbed]) {
    unsigned int firstSegmentIndex = triplets.segmentIndices()[innerTripletIndex][0];
    unsigned int secondSegmentIndex = triplets.segmentIndices()[innerTripletIndex][1];
    unsigned int thirdSegmentIndex = triplets.segmentIndices()[outerTripletIndex][0];
    unsigned int fourthSegmentIndex = triplets.segmentIndices()[outerTripletIndex][1];

    unsigned int firstMDIndex = segments.mdIndices()[firstSegmentIndex][0];
    unsigned int secondMDIndex = segments.mdIndices()[secondSegmentIndex][0];
    unsigned int thirdMDIndex = segments.mdIndices()[secondSegmentIndex][1];
    unsigned int fourthMDIndex = segments.mdIndices()[thirdSegmentIndex][1];
    unsigned int fifthMDIndex = segments.mdIndices()[fourthSegmentIndex][1];

    float x1 = mds.anchorX()[firstMDIndex];
    float x2 = mds.anchorX()[secondMDIndex];
    float x3 = mds.anchorX()[thirdMDIndex];
    float x4 = mds.anchorX()[fourthMDIndex];
    float x5 = mds.anchorX()[fifthMDIndex];

    float y1 = mds.anchorY()[firstMDIndex];
    float y2 = mds.anchorY()[secondMDIndex];
    float y3 = mds.anchorY()[thirdMDIndex];
    float y4 = mds.anchorY()[fourthMDIndex];
    float y5 = mds.anchorY()[fifthMDIndex];

    lst::t5embdnn::runEmbed(acc,
                            mds,
                            firstMDIndex,
                            secondMDIndex,
                            thirdMDIndex,
                            fourthMDIndex,
                            fifthMDIndex,
                            innerRadius,
                            outerRadius,
                            bridgeRadius,
                            triplets.fakeScore()[innerTripletIndex],
                            triplets.promptScore()[innerTripletIndex],
                            triplets.displacedScore()[innerTripletIndex],
                            triplets.fakeScore()[outerTripletIndex],
                            triplets.promptScore()[outerTripletIndex],
                            triplets.displacedScore()[outerTripletIndex],
                            t5Embed);

    // 5 categories for sigmas
    float sigmas2[5], delta1[5], delta2[5], slopes[5];
    bool isFlat[5];

    float xVec[] = {x1, x2, x3, x4, x5};
    float yVec[] = {y1, y2, y3, y4, y5};
    const uint16_t lowerModuleIndices[] = {
        lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5};

    computeSigmasForRegression(acc, modules, lowerModuleIndices, delta1, delta2, slopes, isFlat);
    regressionRadius = computeRadiusUsingRegression(acc,
                                                    Params_T5::kBaseLayers,
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
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool isValidQuintRegion(ModulesConst modules, uint16_t lowerModule) {
    const short layer = modules.layers()[lowerModule];
    const short subdet = modules.subdets()[lowerModule];
    // Quintuplets starting outside these regions are not built.
    return (subdet == Barrel && layer < 3) || (subdet == Endcap && layer <= 1);
  }

  // LS-on-T3 lists by MD: the segments that are the first segment of at least one triplet (the candidate outer LS1s of a
  // T5), listed by their inner MD (in ascending segment index on the serial backend, so walking them and
  // tripletsBySegment visits the triplets of an MD in creation order).
  struct FillLSonT3ByMD {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  MiniDoubletsOccupancyConst mdOccupancy,
                                  SegmentsConst segments,
                                  SegmentsOccupancyConst segmentsOccupancy,
                                  TripletsOccupancyConst tripletsOcc,
                                  TripletsRangesConst tripletsRangesBySegment,
                                  ObjectRangesConst ranges,
                                  unsigned int* __restrict__ nLSonT3ByMD,
                                  unsigned int* __restrict__ offsetLSonT3ByMD,
                                  unsigned int* __restrict__ indexLSonT3) const {
      for (uint16_t lowerModule : cms::alpakatools::independent_groups(acc, modules.nLowerModules())) {
        if (tripletsOcc.nTriplets()[lowerModule] == 0)
          continue;
        const unsigned int firstSegment = ranges.segmentRanges()[lowerModule][0];
        const unsigned int nSegments = segmentsOccupancy.nSegments()[lowerModule];
        for (unsigned int i : cms::alpakatools::independent_group_elements(acc, nSegments)) {
          const unsigned int seg = firstSegment + i;
          if (tripletsRangesBySegment.n()[seg] > 0)
            alpaka::atomicAdd(acc, &nLSonT3ByMD[segments.mdIndices()[seg][0]], 1u, alpaka::hierarchy::Threads{});
        }
        alpaka::syncBlockThreads(acc);
        if (cms::alpakatools::once_per_block(acc)) {
          // A segment is listed at most once, at its inner MD, which is in the segment's own module: the lists of the
          // module's MDs fit in the module's segment range, so the offsets start at its first segment.
          unsigned int offset = firstSegment;
          const unsigned int firstMD = ranges.mdRanges()[lowerModule][0];
          const unsigned int nMDs = mdOccupancy.nMDs()[lowerModule];
          for (unsigned int md = firstMD; md < firstMD + nMDs; ++md) {
            offsetLSonT3ByMD[md] = offset;
            offset += nLSonT3ByMD[md];
            nLSonT3ByMD[md] = 0;
          }
        }
        alpaka::syncBlockThreads(acc);
        for (unsigned int i : cms::alpakatools::independent_group_elements(acc, nSegments)) {
          const unsigned int seg = firstSegment + i;
          if (tripletsRangesBySegment.n()[seg] > 0) {
            const unsigned int md = segments.mdIndices()[seg][0];
            indexLSonT3[offsetLSonT3ByMD[md] +
                        alpaka::atomicAdd(acc, &nLSonT3ByMD[md], 1u, alpaka::hierarchy::Threads{})] = seg;
          }
        }
      }
    }
  };

  // AddSelectedQuintuplets writes the pairs of dense modules first, so the T5 order within a module is that of the
  // serial dense-then-sparse creation (unchanged on the serial backend).
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool isDenseT5Pair(TripletsOccupancyConst tripletsOcc,
                                                    uint16_t lowerModule1,
                                                    uint16_t lowerModule3) {
    return tripletsOcc.nTriplets()[lowerModule1] >= kNTripletThreshold ||
           tripletsOcc.nTriplets()[lowerModule3] >= kNTripletThreshold;
  }

  // Fixed sizes, checked against warpSize in CountTripletConnections: kT5ChunkPairs is the most blockPrefixScan takes
  // with 32-wide warps, not scaled with warpSize, which would make the shared arrays four times larger on 64-wide ones.
  constexpr unsigned int kT5BlockThreads = 256;
  constexpr unsigned int kT5TileInnerT3s = 64;
  constexpr unsigned int kT5ChunkPairs = 1024;

  // Index of the first of the n inclusive prefix sums `ends` that is larger than value (value < ends[n - 1]).
  ALPAKA_FN_ACC ALPAKA_FN_INLINE unsigned int findPrefixSumBin(unsigned int const* ends,
                                                               unsigned int n,
                                                               unsigned int value) {
    unsigned int low = 0, high = n - 1;
    while (low < high) {
      const unsigned int middle = (low + high) / 2;
      if (ends[middle] > value)
        high = middle;
      else
        low = middle + 1;
    }
    return low;
  }

  // The list of the pairs selected by CountTripletConnections (transient, this stage only): the first `capacity` are
  // recorded, `nSelected` counts them all.
  struct SelectedT5Pairs {
    unsigned int* nSelected;
    unsigned int capacity;
    unsigned int* tripletIndices;  // inner and outer triplet of pair i at 2 * i and 2 * i + 1
    float* bridgeRadius;
    float* dnnScore;
  };

  // The remaining cuts of a T5 candidate (inner T3, outer T3) whose outer LS1 passed; a selected one is counted per
  // module and per MD and recorded.
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void selectT5Candidate(TAcc const& acc,
                                                        ModulesConst modules,
                                                        MiniDoubletsConst mds,
                                                        MiniDoubletsT5Counts mdT5Counts,
                                                        SegmentsConst segments,
                                                        TripletsConst triplets,
                                                        TripletCountsConst tripletCountsByMD,
                                                        MiniDoubletsOccupancyConst mdOccupancy,
                                                        const float ptCut,
                                                        unsigned int* __restrict__ outerLS2PassCache,
                                                        unsigned int* __restrict__ moduleT5Count,
                                                        SelectedT5Pairs selected,
                                                        unsigned int innerTripletIndex,
                                                        unsigned int outerTripletIndex,
                                                        float bridgeRadius) {
    const uint16_t lowerModule1 = triplets.lowerModuleIndices()[innerTripletIndex][0];
    // Reuses the decision of an outer LS2 already tested with the same first-inner segment.
    if (!passInnerT3OuterLS2CutsCached(acc,
                                       modules,
                                       mds,
                                       segments,
                                       triplets,
                                       lowerModule1,
                                       innerTripletIndex,
                                       outerTripletIndex,
                                       ptCut,
                                       outerLS2PassCache))
      return;
    float innerRadius, outerRadius, rzChi2, dnnScore;
    if (!runQuintupletSelectionOuter(acc,
                                     modules,
                                     mds,
                                     segments,
                                     triplets,
                                     mdOccupancy,
                                     tripletCountsByMD,
                                     lowerModule1,
                                     innerTripletIndex,
                                     outerTripletIndex,
                                     bridgeRadius,
                                     innerRadius,
                                     outerRadius,
                                     rzChi2,
                                     dnnScore))
      return;
    const unsigned int firstSegmentIndex = triplets.segmentIndices()[innerTripletIndex][0];
    alpaka::atomicAdd(acc, &moduleT5Count[lowerModule1], 1u, alpaka::hierarchy::Blocks{});
    alpaka::atomicAdd(acc,
                      &mdT5Counts.connectedT5s0Max()[segments.mdIndices()[firstSegmentIndex][0]],
                      1u,
                      alpaka::hierarchy::Blocks{});
    alpaka::atomicAdd(acc,
                      &mdT5Counts.connectedT5s1Max()[segments.mdIndices()[firstSegmentIndex][1]],
                      1u,
                      alpaka::hierarchy::Blocks{});
    const unsigned int slot = alpaka::atomicAdd(acc, selected.nSelected, 1u, alpaka::hierarchy::Blocks{});
    if (slot < selected.capacity) {
      selected.tripletIndices[2 * slot] = innerTripletIndex;
      selected.tripletIndices[2 * slot + 1] = outerTripletIndex;
      selected.bridgeRadius[slot] = bridgeRadius;
      selected.dnnScore[slot] = dnnScore;
    }
  }

  // Selects the T5s: counts the selected pairs per module and per MD and records them (createQuintuplets re-runs this
  // kernel with a larger list if more are selected than the list holds).
  // It takes tiles of kT5TileInnerT3s inner T3 slots, one block per tile, and spreads the work of a tile evenly over the
  // threads of the block in two steps, whatever the density:
  // 1. one thread per (inner T3, outer LS1) pair, the outer LS1s of an inner T3 being the LS-on-T3 at its last MD:
  //    the outer LS1 cuts, and the pair's number of outer T3s (the T3s on a passing outer LS1);
  // 2. one thread per candidate (inner T3, outer T3): the outer LS2 cuts and the remaining cuts.
  // The pairs of a tile are taken in chunks of kT5ChunkPairs.
  struct CountTripletConnections {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  MiniDoubletsConst mds,
                                  MiniDoubletsT5Counts mdT5Counts,
                                  SegmentsConst segments,
                                  TripletsConst triplets,
                                  TripletsBySegmentConst tripletsBySegment,
                                  TripletsRangesConst tripletsRangesBySegment,
                                  TripletCountsConst tripletCountsByMD,
                                  MiniDoubletsOccupancyConst mdOccupancy,
                                  unsigned int const* __restrict__ nLSonT3ByMD,
                                  unsigned int const* __restrict__ offsetLSonT3ByMD,
                                  unsigned int const* __restrict__ indexLSonT3,
                                  const unsigned int nTriplets,
                                  const float ptCut,
                                  unsigned int* __restrict__ outerLS2PassCache,
                                  unsigned int* __restrict__ moduleT5Count,
                                  SelectedT5Pairs selected) const {
      const auto& mdIndices = segments.mdIndices();
      const auto& segIdx = triplets.segmentIndices();
      const auto& lmIdx = triplets.lowerModuleIndices();
      const unsigned int nTiles = cms::alpakatools::divide_up_by(nTriplets, kT5TileInnerT3s);

      // Per inner T3 of the tile: its number of outer LS1s and their running sum.
      auto& nOuterLS1s = alpaka::declareSharedVar<unsigned int[kT5TileInnerT3s], __COUNTER__>(acc);
      auto& outerLS1sEnd = alpaka::declareSharedVar<unsigned int[kT5TileInnerT3s], __COUNTER__>(acc);
      // Per (inner T3, outer LS1) pair of the chunk: inner T3, outer LS1, bridge radius, number of outer T3s and their
      // running sum.
      auto& pairInnerT3 = alpaka::declareSharedVar<unsigned int[kT5ChunkPairs], __COUNTER__>(acc);
      auto& pairOuterLS1 = alpaka::declareSharedVar<unsigned int[kT5ChunkPairs], __COUNTER__>(acc);
      auto& pairBridgeRadius = alpaka::declareSharedVar<float[kT5ChunkPairs], __COUNTER__>(acc);
      auto& pairNOuterT3s = alpaka::declareSharedVar<unsigned int[kT5ChunkPairs], __COUNTER__>(acc);
      auto& pairOuterT3sEnd = alpaka::declareSharedVar<unsigned int[kT5ChunkPairs], __COUNTER__>(acc);
      auto& scanWorkspace = alpaka::declareSharedVar<unsigned int[cms::alpakatools::warpSize], __COUNTER__>(acc);
      static_assert(cms::alpakatools::warpSize == 1 ||
                        (kT5ChunkPairs <= cms::alpakatools::warpSize * cms::alpakatools::warpSize &&
                         kT5TileInnerT3s <= cms::alpakatools::warpSize * cms::alpakatools::warpSize &&
                         kT5BlockThreads % cms::alpakatools::warpSize == 0),
                    "blockPrefixScan takes at most warpSize^2 elements and a block of whole warps");

      for (unsigned int tile : cms::alpakatools::independent_groups(acc, nTiles)) {
        const unsigned int firstInnerT3 = tile * kT5TileInnerT3s;
        const unsigned int nInnerT3s =
            nTriplets - firstInnerT3 < kT5TileInnerT3s ? nTriplets - firstInnerT3 : kT5TileInnerT3s;
        for (unsigned int iInnerT3 : cms::alpakatools::independent_group_elements(acc, nInnerT3s)) {
          const unsigned int innerTripletIndex = firstInnerT3 + iInnerT3;
          nOuterLS1s[iInnerT3] = isValidQuintRegion(modules, lmIdx[innerTripletIndex][0])
                                     ? nLSonT3ByMD[mdIndices[segIdx[innerTripletIndex][1]][1]]
                                     : 0;
        }
        alpaka::syncBlockThreads(acc);
        cms::alpakatools::blockPrefixScan(acc, nOuterLS1s, outerLS1sEnd, nInnerT3s, scanWorkspace);
        const unsigned int nPairs = outerLS1sEnd[nInnerT3s - 1];

        for (unsigned int firstPair = 0; firstPair < nPairs; firstPair += kT5ChunkPairs) {
          const unsigned int nChunkPairs = nPairs - firstPair < kT5ChunkPairs ? nPairs - firstPair : kT5ChunkPairs;
          // 1. The outer LS1 cuts, once per (inner T3, outer LS1).
          for (unsigned int iPair : cms::alpakatools::independent_group_elements(acc, nChunkPairs)) {
            const unsigned int iInnerT3 = findPrefixSumBin(outerLS1sEnd, nInnerT3s, firstPair + iPair);
            const unsigned int iOuterLS1 = firstPair + iPair - (outerLS1sEnd[iInnerT3] - nOuterLS1s[iInnerT3]);
            const unsigned int innerTripletIndex = firstInnerT3 + iInnerT3;
            const uint16_t lowerModule1 = lmIdx[innerTripletIndex][0];
            const unsigned int outerLS1Index =
                indexLSonT3[offsetLSonT3ByMD[mdIndices[segIdx[innerTripletIndex][1]][1]] + iOuterLS1];
            pairInnerT3[iPair] = innerTripletIndex;
            pairOuterLS1[iPair] = outerLS1Index;
            // The cap check only skips work: the candidates of a full module are not selected below.
            const bool moduleFull = moduleT5Count[lowerModule1] > kNQuintupletThreshold;
            float dBeta1;
            const bool pass = !moduleFull && passInnerT3OuterLS1Cuts(acc,
                                                                     modules,
                                                                     mds,
                                                                     segments,
                                                                     triplets,
                                                                     lowerModule1,
                                                                     innerTripletIndex,
                                                                     outerLS1Index,
                                                                     dBeta1,
                                                                     ptCut);
            pairNOuterT3s[iPair] = pass ? tripletsRangesBySegment.n()[outerLS1Index] : 0;
            if (pass)
              pairBridgeRadius[iPair] =
                  computeT5BridgeRadius(acc, mds, segments, triplets, innerTripletIndex, outerLS1Index);
          }
          alpaka::syncBlockThreads(acc);
          cms::alpakatools::blockPrefixScan(acc, pairNOuterT3s, pairOuterT3sEnd, nChunkPairs, scanWorkspace);
          const unsigned int nCandidates = pairOuterT3sEnd[nChunkPairs - 1];

          // 2. The other cuts, once per candidate (inner T3, outer T3).
          for (unsigned int iCandidate : cms::alpakatools::independent_group_elements(acc, nCandidates)) {
            const unsigned int iPair = findPrefixSumBin(pairOuterT3sEnd, nChunkPairs, iCandidate);
            const unsigned int innerTripletIndex = pairInnerT3[iPair];
            //asynchronous stop; exact truncation here is not important
            if (moduleT5Count[lmIdx[innerTripletIndex][0]] > kNQuintupletThreshold)
              continue;
            const unsigned int iOuterT3 = iCandidate - (pairOuterT3sEnd[iPair] - pairNOuterT3s[iPair]);
            selectT5Candidate(
                acc,
                modules,
                mds,
                mdT5Counts,
                segments,
                triplets,
                tripletCountsByMD,
                mdOccupancy,
                ptCut,
                outerLS2PassCache,
                moduleT5Count,
                selected,
                innerTripletIndex,
                tripletsBySegment.tripletIndex()[tripletsRangesBySegment.offset()[pairOuterLS1[iPair]] + iOuterT3],
                pairBridgeRadius[iPair]);
          }
          alpaka::syncBlockThreads(acc);
        }
      }
    }
  };

  // Writes the quintuplet of every pair selected by CountTripletConnections at its module's offset, and its by-MD list
  // entries: the dense inner triplets in the first launch, the sparse ones in the second, so that on the serial backend
  // the T5 order within a module is the same as with the creation of dense and sparse modules in two passes.
  struct AddSelectedQuintuplets {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  MiniDoubletsConst mds,
                                  MiniDoubletsT5CountsConst mdT5Counts,
                                  SegmentsConst segments,
                                  Triplets triplets,
                                  TripletsOccupancyConst tripletsOcc,
                                  Quintuplets quintuplets,
                                  QuintupletsOccupancy quintupletsOccupancy,
                                  QuintupletsByMD quintupletsByMD0,
                                  QuintupletsByMD quintupletsByMD1,
                                  QuintupletsRanges quintupletsRangesByMD0,
                                  QuintupletsRanges quintupletsRangesByMD1,
                                  ObjectRanges ranges,
                                  SelectedT5Pairs selected,
                                  const unsigned int nSelected,
#ifdef CUT_VALUE_DEBUG
                                  MiniDoubletsOccupancyConst mdOccupancy,
                                  TripletCountsConst tripletCountsByMD,
                                  const float ptCut,
#endif
                                  const bool densePass) const {
      const auto& mdIndices = segments.mdIndices();
      const auto& segIdx = triplets.segmentIndices();
      const auto& lmIdx = triplets.lowerModuleIndices();
      for (unsigned int i : cms::alpakatools::uniform_elements(acc, nSelected)) {
        const unsigned int innerTripletIndex = selected.tripletIndices[2 * i];
        const unsigned int outerTripletIndex = selected.tripletIndices[2 * i + 1];
        const uint16_t lowerModule1 = lmIdx[innerTripletIndex][0];
        const uint16_t lowerModule3 = lmIdx[innerTripletIndex][2];
        if (isDenseT5Pair(tripletsOcc, lowerModule1, lowerModule3) != densePass)
          continue;

        int totOccupancyQuintuplets = alpaka::atomicAdd(
            acc, &quintupletsOccupancy.totOccupancyQuintuplets()[lowerModule1], 1u, alpaka::hierarchy::Blocks{});
        if (totOccupancyQuintuplets >= ranges.quintupletModuleOccupancy()[lowerModule1]) {
          // A module at the fixed cap kNQuintupletThreshold drops by design; anything else is a counting shortfall.
          alpaka::atomicAdd(acc,
                            ranges.quintupletModuleOccupancy()[lowerModule1] == kNQuintupletThreshold
                                ? &ranges.nQuintupletCapDrops()
                                : &ranges.nQuintupletOverflows(),
                            1u,
                            alpaka::hierarchy::Blocks{});
#ifdef WARNINGS
          printf("Quintuplet excess alert! Module index = %d, Occupancy = %d\n", lowerModule1, totOccupancyQuintuplets);
#endif
          continue;
        }
        const unsigned int quintupletIndex =
            ranges.quintupletModuleIndices()[lowerModule1] +
            alpaka::atomicAdd(acc, &quintupletsOccupancy.nQuintuplets()[lowerModule1], 1u, alpaka::hierarchy::Blocks{});

        auto const ls0Index = segIdx[innerTripletIndex][0];
        auto const md0Index = mdIndices[ls0Index][0];
        auto const quintupletByMD0Local =
            alpaka::atomicAdd(acc, &quintupletsRangesByMD0.n()[md0Index], 1u, alpaka::hierarchy::Blocks{});
        auto quintupletByMD0Index = quintupletByMD0Local + quintupletsRangesByMD0.offset()[md0Index];
        if (quintupletByMD0Local >= mdT5Counts.connectedT5s0Max()[md0Index]) {
          alpaka::atomicSub(acc, &quintupletsRangesByMD0.n()[md0Index], 1u, alpaka::hierarchy::Blocks{});
          quintupletByMD0Index = kInvalidU32Idx;
          alpaka::atomicAdd(acc, &ranges.nT5byMDOverflows(), 1u, alpaka::hierarchy::Blocks{});
        }
        auto const md1Index = mdIndices[ls0Index][1];
        auto const quintupletByMD1Local =
            alpaka::atomicAdd(acc, &quintupletsRangesByMD1.n()[md1Index], 1u, alpaka::hierarchy::Blocks{});
        auto quintupletByMD1Index = quintupletByMD1Local + quintupletsRangesByMD1.offset()[md1Index];
        if (quintupletByMD1Local >= mdT5Counts.connectedT5s1Max()[md1Index]) {
          alpaka::atomicSub(acc, &quintupletsRangesByMD1.n()[md1Index], 1u, alpaka::hierarchy::Blocks{});
          quintupletByMD1Index = kInvalidU32Idx;
          alpaka::atomicAdd(acc, &ranges.nT5byMDOverflows(), 1u, alpaka::hierarchy::Blocks{});
        }

        const uint16_t lowerModule2 = lmIdx[innerTripletIndex][1];
        const uint16_t lowerModule4 = lmIdx[outerTripletIndex][1];
        const uint16_t lowerModule5 = lmIdx[outerTripletIndex][2];
        const float innerRadius = triplets.radius()[innerTripletIndex];
        const float outerRadius = triplets.radius()[outerTripletIndex];
        const float bridgeRadius = selected.bridgeRadius[i];
        float regressionCenterX, regressionCenterY, regressionRadius, chiSquared;
        float t5Embed[Params_T5::kEmbed] = {0.f};
        computeQuintupletExtras(acc,
                                modules,
                                mds,
                                segments,
                                triplets,
                                lowerModule1,
                                lowerModule2,
                                lowerModule3,
                                lowerModule4,
                                lowerModule5,
                                innerTripletIndex,
                                outerTripletIndex,
                                innerRadius,
                                outerRadius,
                                bridgeRadius,
                                regressionCenterX,
                                regressionCenterY,
                                regressionRadius,
                                chiSquared,
                                t5Embed);

        float rzChiSquared = 0.f, dBeta1 = 0.f, dBeta2 = 0.f;
#ifdef CUT_VALUE_DEBUG
        {
          float innerRadiusCut, outerRadiusCut, bridgeRadiusCut, dnnScoreCut;
          runQuintupletSelection(acc,
                                 modules,
                                 mds,
                                 segments,
                                 triplets,
                                 mdOccupancy,
                                 tripletCountsByMD,
                                 lowerModule1,
                                 innerTripletIndex,
                                 outerTripletIndex,
                                 innerRadiusCut,
                                 outerRadiusCut,
                                 bridgeRadiusCut,
                                 rzChiSquared,
                                 dBeta1,
                                 dBeta2,
                                 dnnScoreCut,
                                 ptCut);
        }
#endif
        const auto layer = modules.layers()[lowerModule1];
        // The anchor MD of eta and phi is in the second layer (only layers 1 and 2 start a T5, see isValidQuintRegion).
        const short layer2_adjustment = layer == 1 ? 1 : 0;
        const float phi = mds.anchorPhi()[mdIndices[ls0Index][layer2_adjustment]];
        const float eta = mds.anchorEta()[mdIndices[ls0Index][layer2_adjustment]];
        addQuintupletToMemory(modules,
                              mds,
                              segments,
                              triplets,
                              quintuplets,
                              quintupletsByMD0,
                              quintupletsByMD1,
                              innerTripletIndex,
                              outerTripletIndex,
                              lowerModule1,
                              lowerModule2,
                              lowerModule3,
                              lowerModule4,
                              lowerModule5,
                              innerRadius,
                              bridgeRadius,
                              regressionCenterX,
                              regressionCenterY,
                              regressionRadius,
                              rzChiSquared,
                              chiSquared,
                              dBeta1,
                              dBeta2,
                              eta,
                              phi,
                              layer,
                              quintupletIndex,
                              quintupletByMD0Index,
                              quintupletByMD1Index,
                              t5Embed,
                              selected.dnnScore[i]);
        quintuplets.partOfPT5()[quintupletIndex] = false;
#ifdef CUT_VALUE_DEBUG
        {
          const uint16_t t5Lm[Params_T5::kBaseLayers] = {
              lowerModule1, lowerModule2, lowerModule3, lowerModule4, lowerModule5};
          const unsigned int t5Md[Params_T5::kBaseLayers] = {mdIndices[segIdx[innerTripletIndex][0]][0],
                                                             mdIndices[segIdx[innerTripletIndex][0]][1],
                                                             mdIndices[segIdx[innerTripletIndex][1]][1],
                                                             mdIndices[segIdx[outerTripletIndex][0]][1],
                                                             mdIndices[segIdx[outerTripletIndex][1]][1]};
          const auto t5Feat = computeDnnFeatures<Params_T5::kBaseLayers, 0, 2, 4, 2>(
              acc, modules, mds, mdOccupancy, tripletCountsByMD, t5Lm, t5Md);
          quintuplets.mdDirMeanW()[quintupletIndex] = t5Feat.mdDirMeanW;
          quintuplets.mdDirMaxW()[quintupletIndex] = t5Feat.mdDirMaxW;
          quintuplets.nT3OutMid()[quintupletIndex] = t5Feat.nT3OutMid;
          quintuplets.nT3OutFirst()[quintupletIndex] = t5Feat.nT3OutFirst;
          quintuplets.nMDFirstMod()[quintupletIndex] = t5Feat.nMDFirstMod;
          quintuplets.dcaXY()[quintupletIndex] = t5Feat.dcaXY;
        }
#endif

        triplets.partOfT5()[innerTripletIndex] = true;
        triplets.partOfT5()[outerTripletIndex] = true;
      }
    }
  };

  // T5 slots of a lower module: its selected T5s up to the per-module cap, 0 if it is not eligible.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE unsigned int eligibleT5Slots(ModulesConst modules,
                                                              TripletsOccupancyConst tripletsOcc,
                                                              unsigned int const* __restrict__ moduleT5Count,
                                                              uint16_t lowerModule) {
    if (!isValidQuintRegion(modules, lowerModule) || tripletsOcc.nTriplets()[lowerModule] == 0)
      return 0;
    const unsigned int count = moduleT5Count[lowerModule];
    return count > static_cast<unsigned int>(kNQuintupletThreshold) ? static_cast<unsigned int>(kNQuintupletThreshold)
                                                                    : count;
  }

  // Single block: the eligible modules and their T5 offsets in module order (each thread sums a contiguous chunk of
  // modules, then an exclusive scan of the per-thread sums), so the T5 layout does not depend on the thread scheduling.
  struct CreateEligibleModulesListForQuintuplets {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  TripletsOccupancyConst tripletsOcc,
                                  ObjectRanges ranges,
                                  unsigned int const* __restrict__ moduleT5Count,
                                  MiniDoubletsT5CountsConst mdT5Counts,
                                  MiniDoubletsOccupancyConst mdsOcc,
                                  QuintupletsRanges quintupletsRangesByMD0,
                                  QuintupletsRanges quintupletsRangesByMD1) const {
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));
      constexpr unsigned int kMaxThreads = 1024;
      auto& partialSlots = alpaka::declareSharedVar<unsigned int[kMaxThreads], __COUNTER__>(acc);
      auto& partialModules = alpaka::declareSharedVar<unsigned int[kMaxThreads], __COUNTER__>(acc);
      int& nTotalQuintuplets0x = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      int& nTotalQuintuplets1x = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      const unsigned int nThreads = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
      const unsigned int tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
      ALPAKA_ASSERT_ACC(nThreads <= kMaxThreads);

      const unsigned int nLowerModules = modules.nLowerModules();
      const unsigned int chunk = cms::alpakatools::divide_up_by(nLowerModules, nThreads);
      const unsigned int begin = cms::alpakatools::idx_min(tid * chunk, nLowerModules);
      const unsigned int end = cms::alpakatools::idx_min(begin + chunk, nLowerModules);
      unsigned int slots = 0, eligible = 0;
      for (unsigned int m = begin; m < end; ++m) {
        const unsigned int moduleSlots = eligibleT5Slots(modules, tripletsOcc, moduleT5Count, m);
        slots += moduleSlots;
        eligible += moduleSlots > 0 ? 1 : 0;
      }
      partialSlots[tid] = slots;
      partialModules[tid] = eligible;
      alpaka::syncBlockThreads(acc);
      if (cms::alpakatools::once_per_block(acc)) {
        unsigned int runningSlots = 0, runningModules = 0;
        for (unsigned int t = 0; t < nThreads; ++t) {
          const unsigned int threadSlots = partialSlots[t];
          const unsigned int threadModules = partialModules[t];
          partialSlots[t] = runningSlots;
          partialModules[t] = runningModules;
          runningSlots += threadSlots;
          runningModules += threadModules;
        }
        ranges.nEligibleT5Modules() = static_cast<uint16_t>(runningModules);
        ranges.nTotalQuints() = runningSlots;
        ranges.nQuintupletOverflows() = 0;
        ranges.nT5byMDOverflows() = 0;
        ranges.nQuintupletCapDrops() = 0;
        nTotalQuintuplets0x = 0;
        nTotalQuintuplets1x = 0;
      }
      alpaka::syncBlockThreads(acc);

      unsigned int offset = partialSlots[tid];
      unsigned int eligibleIndex = partialModules[tid];
      for (unsigned int m = begin; m < end; ++m) {
        const unsigned int moduleSlots = eligibleT5Slots(modules, tripletsOcc, moduleT5Count, m);
        if (moduleSlots == 0)
          continue;
        ranges.quintupletModuleIndices()[m] = static_cast<int>(offset);
        ranges.quintupletModuleOccupancy()[m] = static_cast<int>(moduleSlots);
        ranges.indicesOfEligibleT5Modules()[eligibleIndex] = static_cast<uint16_t>(m);
        offset += moduleSlots;
        ++eligibleIndex;
      }

      auto setMDranges = [&](uint16_t const lowerModule,
                             unsigned int const nMDs,
                             unsigned int const* __restrict__ connectedMax,
                             QuintupletsRanges qRanges,
                             int& nTotal) {
        // Sum the real connectivity for MDs in this module
        // There should be no truncation (unlike in nTotalQuintupletsx),
        // else need to keep track which T3's MD can be used in T5 reco
        int connectedInModule = 0;
        const unsigned int firstIdx = ranges.miniDoubletModuleIndices()[lowerModule];
        for (unsigned int idx = 0; idx < nMDs; ++idx) {
          unsigned int mdIndex = firstIdx + idx;
          connectedInModule += connectedMax[mdIndex];
        }

        int nStart = alpaka::atomicAdd(acc, &nTotal, connectedInModule, alpaka::hierarchy::Threads{});

        for (unsigned int idx = 0; idx < nMDs; ++idx) {
          unsigned int mdIndex = firstIdx + idx;
          qRanges.offset()[mdIndex] = nStart;
          qRanges.n()[mdIndex] = 0;
          nStart += connectedMax[mdIndex];
        }
      };
      for (uint16_t lowerModule : cms::alpakatools::uniform_elements(acc, modules.nLowerModules())) {
        unsigned int nMDs = mdsOcc.nMDs()[lowerModule];
        if (nMDs == 0)
          continue;

        setMDranges(
            lowerModule, nMDs, mdT5Counts.connectedT5s0Max().data(), quintupletsRangesByMD0, nTotalQuintuplets0x);
        setMDranges(
            lowerModule, nMDs, mdT5Counts.connectedT5s1Max().data(), quintupletsRangesByMD1, nTotalQuintuplets1x);
      }

      alpaka::syncBlockThreads(acc);
      if (cms::alpakatools::once_per_block(acc)) {
        ranges.nTotalQuintsByMD0() = static_cast<unsigned int>(nTotalQuintuplets0x);
        ranges.nTotalQuintsByMD1() = static_cast<unsigned int>(nTotalQuintuplets1x);
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
