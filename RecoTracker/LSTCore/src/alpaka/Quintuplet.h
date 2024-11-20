#ifndef RecoTracker_LSTCore_src_alpaka_Quintuplet_h
#define RecoTracker_LSTCore_src_alpaka_Quintuplet_h

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "RecoTracker/LSTCore/interface/ObjectRangesSoA.h"
#include "RecoTracker/LSTCore/interface/MiniDoubletsSoA.h"
#include "RecoTracker/LSTCore/interface/SegmentsSoA.h"
#include "RecoTracker/LSTCore/interface/TripletsSoA.h"
#include "RecoTracker/LSTCore/interface/QuintupletsSoA.h"
#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/ModulesSoA.h"
#include "RecoTracker/LSTCore/interface/EndcapGeometry.h"
#include "RecoTracker/LSTCore/interface/ObjectRangesSoA.h"

#include "NeuralNetwork.h"
#include "Hit.h"
#include "Triplet.h"  // FIXME: need to refactor common functions to a common place

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool checkIntervalOverlap(float firstMin,
                                                           float firstMax,
                                                           float secondMin,
                                                           float secondMax) {
    return ((firstMin <= secondMin) && (secondMin < firstMax)) || ((secondMin < firstMin) && (firstMin < secondMax));
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addQuintupletToMemory(TripletsConst triplets,
                                                            Quintuplets quintuplets,
                                                            unsigned int innerTripletIndex,
                                                            unsigned int outerTripletIndex,
                                                            uint16_t lowerModule1,
                                                            uint16_t lowerModule2,
                                                            uint16_t lowerModule3,
                                                            uint16_t lowerModule4,
                                                            uint16_t lowerModule5,
                                                            float innerRadius,
                                                            float bridgeRadius,
                                                            float outerRadius,
                                                            float regressionG,
                                                            float regressionF,
                                                            float regressionRadius,
                                                            float rzChiSquared,
                                                            float rPhiChiSquared,
                                                            float nonAnchorChiSquared,
                                                            float pt,
                                                            float eta,
                                                            float phi,
                                                            float scores,
                                                            uint8_t layer,
                                                            unsigned int quintupletIndex,
                                                            bool tightCutFlag) {
    quintuplets.tripletIndices()[quintupletIndex][0] = innerTripletIndex;
    quintuplets.tripletIndices()[quintupletIndex][1] = outerTripletIndex;

    quintuplets.lowerModuleIndices()[quintupletIndex][0] = lowerModule1;
    quintuplets.lowerModuleIndices()[quintupletIndex][1] = lowerModule2;
    quintuplets.lowerModuleIndices()[quintupletIndex][2] = lowerModule3;
    quintuplets.lowerModuleIndices()[quintupletIndex][3] = lowerModule4;
    quintuplets.lowerModuleIndices()[quintupletIndex][4] = lowerModule5;
    quintuplets.innerRadius()[quintupletIndex] = __F2H(innerRadius);
    quintuplets.outerRadius()[quintupletIndex] = __F2H(outerRadius);
    quintuplets.pt()[quintupletIndex] = __F2H(pt);
    quintuplets.eta()[quintupletIndex] = __F2H(eta);
    quintuplets.phi()[quintupletIndex] = __F2H(phi);
    quintuplets.score_rphisum()[quintupletIndex] = __F2H(scores);
    quintuplets.isDup()[quintupletIndex] = 0;
    quintuplets.tightCutFlag()[quintupletIndex] = tightCutFlag;
    quintuplets.regressionRadius()[quintupletIndex] = regressionRadius;
    quintuplets.regressionG()[quintupletIndex] = regressionG;
    quintuplets.regressionF()[quintupletIndex] = regressionF;
    quintuplets.logicalLayers()[quintupletIndex][0] = triplets.logicalLayers()[innerTripletIndex][0];
    quintuplets.logicalLayers()[quintupletIndex][1] = triplets.logicalLayers()[innerTripletIndex][1];
    quintuplets.logicalLayers()[quintupletIndex][2] = triplets.logicalLayers()[innerTripletIndex][2];
    quintuplets.logicalLayers()[quintupletIndex][3] = triplets.logicalLayers()[outerTripletIndex][1];
    quintuplets.logicalLayers()[quintupletIndex][4] = triplets.logicalLayers()[outerTripletIndex][2];

    quintuplets.hitIndices()[quintupletIndex][0] = triplets.hitIndices()[innerTripletIndex][0];
    quintuplets.hitIndices()[quintupletIndex][1] = triplets.hitIndices()[innerTripletIndex][1];
    quintuplets.hitIndices()[quintupletIndex][2] = triplets.hitIndices()[innerTripletIndex][2];
    quintuplets.hitIndices()[quintupletIndex][3] = triplets.hitIndices()[innerTripletIndex][3];
    quintuplets.hitIndices()[quintupletIndex][4] = triplets.hitIndices()[innerTripletIndex][4];
    quintuplets.hitIndices()[quintupletIndex][5] = triplets.hitIndices()[innerTripletIndex][5];
    quintuplets.hitIndices()[quintupletIndex][6] = triplets.hitIndices()[outerTripletIndex][2];
    quintuplets.hitIndices()[quintupletIndex][7] = triplets.hitIndices()[outerTripletIndex][3];
    quintuplets.hitIndices()[quintupletIndex][8] = triplets.hitIndices()[outerTripletIndex][4];
    quintuplets.hitIndices()[quintupletIndex][9] = triplets.hitIndices()[outerTripletIndex][5];
    quintuplets.bridgeRadius()[quintupletIndex] = bridgeRadius;
    quintuplets.rzChiSquared()[quintupletIndex] = rzChiSquared;
    quintuplets.chiSquared()[quintupletIndex] = rPhiChiSquared;
    quintuplets.nonAnchorChiSquared()[quintupletIndex] = nonAnchorChiSquared;
  }

  //90% constraint
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passChiSquaredConstraint(ModulesConst modules,
                                                               uint16_t lowerModuleIndex1,
                                                               uint16_t lowerModuleIndex2,
                                                               uint16_t lowerModuleIndex3,
                                                               uint16_t lowerModuleIndex4,
                                                               uint16_t lowerModuleIndex5,
                                                               float chiSquared) {
    // Using lstLayer numbering convention defined in ModuleMethods.h
    const int layer1 = modules.lstLayers()[lowerModuleIndex1];
    const int layer2 = modules.lstLayers()[lowerModuleIndex2];
    const int layer3 = modules.lstLayers()[lowerModuleIndex3];
    const int layer4 = modules.lstLayers()[lowerModuleIndex4];
    const int layer5 = modules.lstLayers()[lowerModuleIndex5];

    if (layer1 == 7 and layer2 == 8 and layer3 == 9) {
      if (layer4 == 10 and layer5 == 11) {
        return chiSquared < 0.01788f;
      } else if (layer4 == 10 and layer5 == 16) {
        return chiSquared < 0.04725f;
      } else if (layer4 == 15 and layer5 == 16) {
        return chiSquared < 0.04725f;
      }
    } else if (layer1 == 1 and layer2 == 7 and layer3 == 8) {
      if (layer4 == 9 and layer5 == 10) {
        return chiSquared < 0.01788f;
      } else if (layer4 == 9 and layer5 == 15) {
        return chiSquared < 0.08234f;
      }
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 7) {
      if (layer4 == 8 and layer5 == 9) {
        return chiSquared < 0.02360f;
      } else if (layer4 == 8 and layer5 == 14) {
        return chiSquared < 0.07167f;
      } else if (layer4 == 13 and layer5 == 14) {
        return chiSquared < 0.08234f;
      }
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 3) {
      if (layer4 == 7 and layer5 == 8) {
        return chiSquared < 0.01026f;
      } else if (layer4 == 7 and layer5 == 13) {
        return chiSquared < 0.06238f;
      } else if (layer4 == 12 and layer5 == 13) {
        return chiSquared < 0.06238f;
      }
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 3 and layer4 == 4) {
      if (layer5 == 5) {
        return chiSquared < 0.04725f;
      } else if (layer5 == 12) {
        return chiSquared < 0.09461f;
      }
    } else if (layer1 == 2 and layer2 == 7 and layer3 == 8) {
      if (layer4 == 9 and layer5 == 10) {
        return chiSquared < 0.00512f;
      }
      if (layer4 == 9 and layer5 == 15) {
        return chiSquared < 0.04112f;
      } else if (layer4 == 14 and layer5 == 15) {
        return chiSquared < 0.06238f;
      }
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 7) {
      if (layer4 == 8 and layer5 == 14) {
        return chiSquared < 0.07167f;
      } else if (layer4 == 13 and layer5 == 14) {
        return chiSquared < 0.06238f;
      }
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 4) {
      if (layer4 == 5 and layer5 == 6) {
        return chiSquared < 0.08234f;
      } else if (layer4 == 5 and layer5 == 12) {
        return chiSquared < 0.10870f;
      } else if (layer4 == 12 and layer5 == 13) {
        return chiSquared < 0.10870f;
      }
    } else if (layer1 == 3 and layer2 == 7 and layer3 == 8 and layer4 == 14 and layer5 == 15) {
      return chiSquared < 0.09461f;
    } else if (layer1 == 3 and layer2 == 4 and layer3 == 5 and layer4 == 12 and layer5 == 13) {
      return chiSquared < 0.09461f;
    }

    return true;
  }

  //bounds can be found at http://uaf-10.t2.ucsd.edu/~bsathian/SDL/T5_RZFix/t5_rz_thresholds.txt
  template <typename TAcc>
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
                                                         float f,
                                                         bool& tightCutFlag) {
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

    // when building T5, apply 99% chi2 cuts as default, and add to pT5 collection. But when adding T5 to TC collections, apply 95% cut to reduce the fake rate
    tightCutFlag = false;
    // The category numbers are related to module regions and layers, decoding of the region numbers can be found here in slide 2 table. https://github.com/SegmentLinking/TrackLooper/files/11420927/part.2.pdf
    // The commented numbers after each case is the region code, and can look it up from the table to see which category it belongs to. For example, //0 means T5 built with Endcap 1,2,3,4,5 ps modules
    if (layer1 == 7 and layer2 == 8 and layer3 == 9 and layer4 == 10 and layer5 == 11)  //0
    {
      if (rzChiSquared < 94.470f)
        tightCutFlag = true;
      return true;
    } else if (layer1 == 7 and layer2 == 8 and layer3 == 9 and layer4 == 10 and layer5 == 16)  //1
    {
      if (rzChiSquared < 22.099f)
        tightCutFlag = true;
      return rzChiSquared < 37.956f;
    } else if (layer1 == 7 and layer2 == 8 and layer3 == 9 and layer4 == 15 and layer5 == 16)  //2
    {
      if (rzChiSquared < 7.992f)
        tightCutFlag = true;
      return rzChiSquared < 11.622f;
    } else if (layer1 == 1 and layer2 == 7 and layer3 == 8 and layer4 == 9) {
      if (layer5 == 10)  //3
      {
        if (rzChiSquared < 111.390f)
          tightCutFlag = true;
        return true;
      }
      if (layer5 == 15)  //4
      {
        if (rzChiSquared < 18.351f)
          tightCutFlag = true;
        return rzChiSquared < 37.941f;
      }
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 7) {
      if (layer4 == 8 and layer5 == 9)  //5
      {
        if (rzChiSquared < 116.148f)
          tightCutFlag = true;
        return true;
      }
      if (layer4 == 8 and layer5 == 14)  //6
      {
        if (rzChiSquared < 19.352f)
          tightCutFlag = true;
        return rzChiSquared < 52.561f;
      } else if (layer4 == 13 and layer5 == 14)  //7
      {
        if (rzChiSquared < 10.392f)
          tightCutFlag = true;
        return rzChiSquared < 13.76f;
      }
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 3) {
      if (layer4 == 7 and layer5 == 8)  //8
      {
        if (rzChiSquared < 27.824f)
          tightCutFlag = true;
        return rzChiSquared < 44.247f;
      } else if (layer4 == 7 and layer5 == 13)  //9
      {
        if (rzChiSquared < 18.145f)
          tightCutFlag = true;
        return rzChiSquared < 33.752f;
      } else if (layer4 == 12 and layer5 == 13)  //10
      {
        if (rzChiSquared < 13.308f)
          tightCutFlag = true;
        return rzChiSquared < 21.213f;
      } else if (layer4 == 4 and layer5 == 5)  //11
      {
        if (rzChiSquared < 15.627f)
          tightCutFlag = true;
        return rzChiSquared < 29.035f;
      } else if (layer4 == 4 and layer5 == 12)  //12
      {
        if (rzChiSquared < 14.64f)
          tightCutFlag = true;
        return rzChiSquared < 23.037f;
      }
    } else if (layer1 == 2 and layer2 == 7 and layer3 == 8) {
      if (layer4 == 9 and layer5 == 15)  //14
      {
        if (rzChiSquared < 24.662f)
          tightCutFlag = true;
        return rzChiSquared < 41.036f;
      } else if (layer4 == 14 and layer5 == 15)  //15
      {
        if (rzChiSquared < 8.866f)
          tightCutFlag = true;
        return rzChiSquared < 14.092f;
      }
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 7) {
      if (layer4 == 8 and layer5 == 14)  //16
      {
        if (rzChiSquared < 23.730f)
          tightCutFlag = true;
        return rzChiSquared < 23.748f;
      }
      if (layer4 == 13 and layer5 == 14)  //17
      {
        if (rzChiSquared < 10.772f)
          tightCutFlag = true;
        return rzChiSquared < 17.945f;
      }
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 4) {
      if (layer4 == 5 and layer5 == 6)  //18
      {
        if (rzChiSquared < 6.065f)
          tightCutFlag = true;
        return rzChiSquared < 8.803f;
      } else if (layer4 == 5 and layer5 == 12)  //19
      {
        if (rzChiSquared < 5.693f)
          tightCutFlag = true;
        return rzChiSquared < 7.930f;
      }

      else if (layer4 == 12 and layer5 == 13)  //20
      {
        if (rzChiSquared < 5.473f)
          tightCutFlag = true;
        return rzChiSquared < 7.626f;
      }
    }
    return true;
  }

  template <typename TAcc>
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

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void computeErrorInRadius(TAcc const& acc,
                                                           float* x1Vec,
                                                           float* y1Vec,
                                                           float* x2Vec,
                                                           float* y2Vec,
                                                           float* x3Vec,
                                                           float* y3Vec,
                                                           float& minimumRadius,
                                                           float& maximumRadius) {
    //brute force
    float candidateRadius;
    float g, f;
    minimumRadius = kVerticalModuleSlope;
    maximumRadius = 0.f;
    for (size_t i = 0; i < 3; i++) {
      float x1 = x1Vec[i];
      float y1 = y1Vec[i];
      for (size_t j = 0; j < 3; j++) {
        float x2 = x2Vec[j];
        float y2 = y2Vec[j];
        for (size_t k = 0; k < 3; k++) {
          float x3 = x3Vec[k];
          float y3 = y3Vec[k];
          candidateRadius = computeRadiusFromThreeAnchorHits(acc, x1, y1, x2, y2, x3, y3, g, f);
          maximumRadius = alpaka::math::max(acc, candidateRadius, maximumRadius);
          minimumRadius = alpaka::math::min(acc, candidateRadius, minimumRadius);
        }
      }
    }
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool matchRadiiBBBEE12378(TAcc const& acc,
                                                           float innerRadius,
                                                           float bridgeRadius,
                                                           float outerRadius,
                                                           float bridgeRadiusMin2S,
                                                           float bridgeRadiusMax2S) {
    float innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax;

    float innerInvRadiusErrorBound = 0.178f;
    float bridgeInvRadiusErrorBound = 0.507f;

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
    bridgeInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

    return checkIntervalOverlap(innerInvRadiusMin,
                                innerInvRadiusMax,
                                alpaka::math::min(acc, bridgeInvRadiusMin, 1.0f / bridgeRadiusMax2S),
                                alpaka::math::max(acc, bridgeInvRadiusMax, 1.0f / bridgeRadiusMin2S));
  }

  /*bounds for high Pt taken from : http://uaf-10.t2.ucsd.edu/~bsathian/SDL/T5_efficiency/efficiencies/new_efficiencies/efficiencies_20210513_T5_recovering_high_Pt_efficiencies/highE_radius_matching/highE_bounds.txt */
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool matchRadiiBBBBB(TAcc const& acc,
                                                      float innerRadius,
                                                      float bridgeRadius,
                                                      float outerRadius) {
    float innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax;

    float innerInvRadiusErrorBound = 0.1512f;
    float bridgeInvRadiusErrorBound = 0.1781f;

    if (innerRadius * k2Rinv1GeVf > 1.f) {
      innerInvRadiusErrorBound = 0.4449f;
      bridgeInvRadiusErrorBound = 0.4033f;
    }

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
    bridgeInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

    return checkIntervalOverlap(innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax);
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool matchRadiiBBBBE(TAcc const& acc,
                                                      float innerRadius,
                                                      float bridgeRadius,
                                                      float outerRadius) {
    float innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax;

    float innerInvRadiusErrorBound = 0.1781f;
    float bridgeInvRadiusErrorBound = 0.2167f;

    if (innerRadius * k2Rinv1GeVf > 1.f) {
      innerInvRadiusErrorBound = 0.4750f;
      bridgeInvRadiusErrorBound = 0.3903f;
    }

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
    bridgeInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

    return checkIntervalOverlap(innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax);
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool matchRadiiBBBEE23478(TAcc const& acc,
                                                           float innerRadius,
                                                           float bridgeRadius,
                                                           float outerRadius,
                                                           float bridgeRadiusMin2S,
                                                           float bridgeRadiusMax2S) {
    float innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax;

    float innerInvRadiusErrorBound = 0.2097f;
    float bridgeInvRadiusErrorBound = 0.8557f;

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
    bridgeInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

    return checkIntervalOverlap(innerInvRadiusMin,
                                innerInvRadiusMax,
                                alpaka::math::min(acc, bridgeInvRadiusMin, 1.0f / bridgeRadiusMax2S),
                                alpaka::math::max(acc, bridgeInvRadiusMax, 1.0f / bridgeRadiusMin2S));
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool matchRadiiBBBEE34578(TAcc const& acc,
                                                           float innerRadius,
                                                           float bridgeRadius,
                                                           float outerRadius,
                                                           float bridgeRadiusMin2S,
                                                           float bridgeRadiusMax2S) {
    float innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax;

    float innerInvRadiusErrorBound = 0.066f;
    float bridgeInvRadiusErrorBound = 0.617f;

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
    bridgeInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

    return checkIntervalOverlap(innerInvRadiusMin,
                                innerInvRadiusMax,
                                alpaka::math::min(acc, bridgeInvRadiusMin, 1.0f / bridgeRadiusMax2S),
                                alpaka::math::max(acc, bridgeInvRadiusMax, 1.0f / bridgeRadiusMin2S));
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool matchRadiiBBEEE(TAcc const& acc,
                                                      float innerRadius,
                                                      float bridgeRadius,
                                                      float outerRadius,
                                                      float bridgeRadiusMin2S,
                                                      float bridgeRadiusMax2S) {
    float innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax;

    float innerInvRadiusErrorBound = 0.6376f;
    float bridgeInvRadiusErrorBound = 2.1381f;

    if (innerRadius * k2Rinv1GeVf > 1.f)  //as good as no selections!
    {
      innerInvRadiusErrorBound = 12.9173f;
      bridgeInvRadiusErrorBound = 5.1700f;
    }

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
    bridgeInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

    return checkIntervalOverlap(innerInvRadiusMin,
                                innerInvRadiusMax,
                                alpaka::math::min(acc, bridgeInvRadiusMin, 1.0f / bridgeRadiusMax2S),
                                alpaka::math::max(acc, bridgeInvRadiusMax, 1.0f / bridgeRadiusMin2S));
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool matchRadiiBEEEE(TAcc const& acc,
                                                      float innerRadius,
                                                      float bridgeRadius,
                                                      float outerRadius,
                                                      float innerRadiusMin2S,
                                                      float innerRadiusMax2S,
                                                      float bridgeRadiusMin2S,
                                                      float bridgeRadiusMax2S) {
    float innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax;

    float innerInvRadiusErrorBound = 1.9382f;
    float bridgeInvRadiusErrorBound = 3.7280f;

    if (innerRadius * k2Rinv1GeVf > 1.f) {
      innerInvRadiusErrorBound = 23.2713f;
      bridgeInvRadiusErrorBound = 21.7980f;
    }

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
    bridgeInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

    return checkIntervalOverlap(alpaka::math::min(acc, innerInvRadiusMin, 1.0f / innerRadiusMax2S),
                                alpaka::math::max(acc, innerInvRadiusMax, 1.0f / innerRadiusMin2S),
                                alpaka::math::min(acc, bridgeInvRadiusMin, 1.0f / bridgeRadiusMax2S),
                                alpaka::math::max(acc, bridgeInvRadiusMax, 1.0f / bridgeRadiusMin2S));
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool matchRadiiEEEEE(TAcc const& acc,
                                                      float innerRadius,
                                                      float bridgeRadius,
                                                      float outerRadius,
                                                      float innerRadiusMin2S,
                                                      float innerRadiusMax2S,
                                                      float bridgeRadiusMin2S,
                                                      float bridgeRadiusMax2S) {
    float innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax;

    float innerInvRadiusErrorBound = 1.9382f;
    float bridgeInvRadiusErrorBound = 2.2091f;

    if (innerRadius * k2Rinv1GeVf > 1.f) {
      innerInvRadiusErrorBound = 22.5226f;
      bridgeInvRadiusErrorBound = 21.0966f;
    }

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
    bridgeInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

    return checkIntervalOverlap(alpaka::math::min(acc, innerInvRadiusMin, 1.0f / innerRadiusMax2S),
                                alpaka::math::max(acc, innerInvRadiusMax, 1.0f / innerRadiusMin2S),
                                alpaka::math::min(acc, bridgeInvRadiusMin, 1.0f / bridgeRadiusMax2S),
                                alpaka::math::max(acc, bridgeInvRadiusMax, 1.0f / bridgeRadiusMin2S));
  }

  template <typename TAcc>
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

  template <typename TAcc>
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

      absArctanSlope = ((slopes[i] != kVerticalModuleSlope) ? alpaka::math::abs(acc, alpaka::math::atan(acc, slopes[i]))
                                                            : kPi / 2.f);

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

  template <typename TAcc>
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
      absArctanSlope = ((slopes[i] != kVerticalModuleSlope) ? alpaka::math::abs(acc, alpaka::math::atan(acc, slopes[i]))
                                                            : kPi / 2.f);
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

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void runDeltaBetaIterations(TAcc const& acc,
                                                             float& betaIn,
                                                             float& betaOut,
                                                             float betaAv,
                                                             float& pt_beta,
                                                             float sdIn_dr,
                                                             float sdOut_dr,
                                                             float dr,
                                                             float lIn) {
    if (lIn == 0) {
      betaOut += alpaka::math::copysign(
          acc,
          alpaka::math::asin(
              acc, alpaka::math::min(acc, sdOut_dr * k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), kSinAlphaMax)),
          betaOut);
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
      betaAv = 0.5f * (betaInUpd + betaOutUpd);

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
      betaAv = (alpaka::math::abs(acc, betaOut) > 0.2f * alpaka::math::abs(acc, betaIn))
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

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuintupletDefaultAlgoBBBB(TAcc const& acc,
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
                                                                   unsigned int fourthMDIndex) {
    bool isPS_InLo = (modules.moduleType()[innerInnerLowerModuleIndex] == PS);
    bool isPS_OutLo = (modules.moduleType()[outerInnerLowerModuleIndex] == PS);

    float rt_InLo = mds.anchorRt()[firstMDIndex];
    float rt_InOut = mds.anchorRt()[secondMDIndex];
    float rt_OutLo = mds.anchorRt()[thirdMDIndex];

    float z_InLo = mds.anchorZ()[firstMDIndex];
    float z_InOut = mds.anchorZ()[secondMDIndex];
    float z_OutLo = mds.anchorZ()[thirdMDIndex];

    float alpha1GeV_OutLo =
        alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * k2Rinv1GeVf / ptCut, kSinAlphaMax));

    float rtRatio_OutLoInLo = rt_OutLo / rt_InLo;  // Outer segment beginning rt divided by inner segment beginning rt;
    float dzDrtScale =
        alpaka::math::tan(acc, alpha1GeV_OutLo) / alpha1GeV_OutLo;  // The track can bend in r-z plane slightly
    float zpitch_InLo = (isPS_InLo ? kPixelPSZpitch : kStrip2SZpitch);
    float zpitch_OutLo = (isPS_OutLo ? kPixelPSZpitch : kStrip2SZpitch);

    float zHi = z_InLo + (z_InLo + kDeltaZLum) * (rtRatio_OutLoInLo - 1.f) * (z_InLo < 0.f ? 1.f : dzDrtScale) +
                (zpitch_InLo + zpitch_OutLo);
    float zLo = z_InLo + (z_InLo - kDeltaZLum) * (rtRatio_OutLoInLo - 1.f) * (z_InLo > 0.f ? 1.f : dzDrtScale) -
                (zpitch_InLo + zpitch_OutLo);

    //Cut 1 - z compatibility
    if ((z_OutLo < zLo) || (z_OutLo > zHi))
      return false;

    float drt_OutLo_InLo = (rt_OutLo - rt_InLo);
    float r3_InLo = alpaka::math::sqrt(acc, z_InLo * z_InLo + rt_InLo * rt_InLo);
    float drt_InSeg = rt_InOut - rt_InLo;
    float dz_InSeg = z_InOut - z_InLo;
    float dr3_InSeg = alpaka::math::sqrt(acc, rt_InOut * rt_InOut + z_InOut * z_InOut) -
                      alpaka::math::sqrt(acc, rt_InLo * rt_InLo + z_InLo * z_InLo);

    float coshEta = dr3_InSeg / drt_InSeg;
    float dzErr = (zpitch_InLo + zpitch_OutLo) * (zpitch_InLo + zpitch_OutLo) * 2.f;

    float thetaMuls2 = (kMulsInGeV * kMulsInGeV) * (0.1f + 0.2f * (rt_OutLo - rt_InLo) / 50.f) * (r3_InLo / rt_InLo);
    float muls2 = thetaMuls2 * 9.f / (ptCut * ptCut) * 16.f;
    dzErr += muls2 * drt_OutLo_InLo * drt_OutLo_InLo / 3.f * coshEta * coshEta;
    dzErr = alpaka::math::sqrt(acc, dzErr);

    // Constructing upper and lower bound
    const float dzMean = dz_InSeg / drt_InSeg * drt_OutLo_InLo;
    const float zWindow =
        dzErr / drt_InSeg * drt_OutLo_InLo +
        (zpitch_InLo + zpitch_OutLo);  //FIXME for ptCut lower than ~0.8 need to add curv path correction
    float zLoPointed = z_InLo + dzMean * (z_InLo > 0.f ? 1.f : dzDrtScale) - zWindow;
    float zHiPointed = z_InLo + dzMean * (z_InLo < 0.f ? 1.f : dzDrtScale) + zWindow;

    // Cut #2: Pointed Z (Inner segment two MD points to outer segment inner MD)
    if ((z_OutLo < zLoPointed) || (z_OutLo > zHiPointed))
      return false;

    float pvOffset = 0.1f / rt_OutLo;
    float dPhiCut = alpha1GeV_OutLo + alpaka::math::sqrt(acc, muls2 + pvOffset * pvOffset);

    float deltaPhiPos = phi_mpi_pi(acc, mds.anchorPhi()[fourthMDIndex] - mds.anchorPhi()[secondMDIndex]);
    // Cut #3: FIXME:deltaPhiPos can be tighter
    if (alpaka::math::abs(acc, deltaPhiPos) > dPhiCut)
      return false;

    float midPointX = 0.5f * (mds.anchorX()[firstMDIndex] + mds.anchorX()[thirdMDIndex]);
    float midPointY = 0.5f * (mds.anchorY()[firstMDIndex] + mds.anchorY()[thirdMDIndex]);
    float diffX = mds.anchorX()[thirdMDIndex] - mds.anchorX()[firstMDIndex];
    float diffY = mds.anchorY()[thirdMDIndex] - mds.anchorY()[firstMDIndex];

    float dPhi = deltaPhi(acc, midPointX, midPointY, diffX, diffY);

    // Cut #4: deltaPhiChange
    if (alpaka::math::abs(acc, dPhi) > dPhiCut)
      return false;

    // First obtaining the raw betaIn and betaOut values without any correction and just purely based on the mini-doublet hit positions
    float alpha_InLo = __H2F(segments.dPhiChanges()[innerSegmentIndex]);
    float alpha_OutLo = __H2F(segments.dPhiChanges()[outerSegmentIndex]);

    bool isEC_lastLayer = modules.subdets()[outerOuterLowerModuleIndex] == Endcap and
                          modules.moduleType()[outerOuterLowerModuleIndex] == TwoS;

    float alpha_OutUp, alpha_OutUp_highEdge, alpha_OutUp_lowEdge;

    alpha_OutUp = phi_mpi_pi(acc,
                             phi(acc,
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

    float betaIn = alpha_InLo - phi_mpi_pi(acc, phi(acc, tl_axis_x, tl_axis_y) - mds.anchorPhi()[firstMDIndex]);

    float betaInRHmin = betaIn;
    float betaInRHmax = betaIn;
    float betaOut = -alpha_OutUp + phi_mpi_pi(acc, phi(acc, tl_axis_x, tl_axis_y) - mds.anchorPhi()[fourthMDIndex]);

    float betaOutRHmin = betaOut;
    float betaOutRHmax = betaOut;

    if (isEC_lastLayer) {
      alpha_OutUp_highEdge = phi_mpi_pi(acc,
                                        phi(acc,
                                            mds.anchorHighEdgeX()[fourthMDIndex] - mds.anchorX()[thirdMDIndex],
                                            mds.anchorHighEdgeY()[fourthMDIndex] - mds.anchorY()[thirdMDIndex]) -
                                            mds.anchorHighEdgePhi()[fourthMDIndex]);
      alpha_OutUp_lowEdge = phi_mpi_pi(acc,
                                       phi(acc,
                                           mds.anchorLowEdgeX()[fourthMDIndex] - mds.anchorX()[thirdMDIndex],
                                           mds.anchorLowEdgeY()[fourthMDIndex] - mds.anchorY()[thirdMDIndex]) -
                                           mds.anchorLowEdgePhi()[fourthMDIndex]);

      tl_axis_highEdge_x = mds.anchorHighEdgeX()[fourthMDIndex] - mds.anchorX()[firstMDIndex];
      tl_axis_highEdge_y = mds.anchorHighEdgeY()[fourthMDIndex] - mds.anchorY()[firstMDIndex];
      tl_axis_lowEdge_x = mds.anchorLowEdgeX()[fourthMDIndex] - mds.anchorX()[firstMDIndex];
      tl_axis_lowEdge_y = mds.anchorLowEdgeY()[fourthMDIndex] - mds.anchorY()[firstMDIndex];

      betaOutRHmin =
          -alpha_OutUp_highEdge +
          phi_mpi_pi(acc, phi(acc, tl_axis_highEdge_x, tl_axis_highEdge_y) - mds.anchorHighEdgePhi()[fourthMDIndex]);
      betaOutRHmax =
          -alpha_OutUp_lowEdge +
          phi_mpi_pi(acc, phi(acc, tl_axis_lowEdge_x, tl_axis_lowEdge_y) - mds.anchorLowEdgePhi()[fourthMDIndex]);
    }

    //beta computation
    float drt_tl_axis = alpaka::math::sqrt(acc, tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);

    float corrF = 1.f;
    //innerOuterAnchor - innerInnerAnchor
    const float rt_InSeg = alpaka::math::sqrt(acc,
                                              (mds.anchorX()[secondMDIndex] - mds.anchorX()[firstMDIndex]) *
                                                      (mds.anchorX()[secondMDIndex] - mds.anchorX()[firstMDIndex]) +
                                                  (mds.anchorY()[secondMDIndex] - mds.anchorY()[firstMDIndex]) *
                                                      (mds.anchorY()[secondMDIndex] - mds.anchorY()[firstMDIndex]));
    float betaInCut =
        alpaka::math::asin(
            acc, alpaka::math::min(acc, (-rt_InSeg * corrF + drt_tl_axis) * k2Rinv1GeVf / ptCut, kSinAlphaMax)) +
        (0.02f / drt_InSeg);

    //Cut #5: first beta cut
    if (alpaka::math::abs(acc, betaInRHmin) >= betaInCut)
      return false;

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

    runDeltaBetaIterations(acc, betaIn, betaOut, betaAv, pt_beta, rt_InSeg, sdOut_dr, drt_tl_axis, lIn);

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

    const float dBetaRIn2 = 0;  // TODO-RH
    float dBetaROut = 0;
    if (isEC_lastLayer) {
      dBetaROut = (alpaka::math::sqrt(acc,
                                      mds.anchorHighEdgeX()[fourthMDIndex] * mds.anchorHighEdgeX()[fourthMDIndex] +
                                          mds.anchorHighEdgeY()[fourthMDIndex] * mds.anchorHighEdgeY()[fourthMDIndex]) -
                   alpaka::math::sqrt(acc,
                                      mds.anchorLowEdgeX()[fourthMDIndex] * mds.anchorLowEdgeX()[fourthMDIndex] +
                                          mds.anchorLowEdgeY()[fourthMDIndex] * mds.anchorLowEdgeY()[fourthMDIndex])) *
                  sinDPhi / drt_tl_axis;
    }

    const float dBetaROut2 = dBetaROut * dBetaROut;

    float betaOutCut =
        alpaka::math::asin(acc, alpaka::math::min(acc, drt_tl_axis * k2Rinv1GeVf / ptCut, kSinAlphaMax)) +
        (0.02f / sdOut_d) + alpaka::math::sqrt(acc, dBetaLum2 + dBetaMuls2);

    //Cut #6: The real beta cut
    if (alpaka::math::abs(acc, betaOut) >= betaOutCut)
      return false;

    float dBetaRes = 0.02f / alpaka::math::min(acc, sdOut_d, drt_InSeg);
    float dBetaCut2 =
        (dBetaRes * dBetaRes * 2.0f + dBetaMuls2 + dBetaLum2 + dBetaRIn2 + dBetaROut2 +
         0.25f *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)) *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)));

    float dBeta = betaIn - betaOut;
    return dBeta * dBeta <= dBetaCut2;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuintupletDefaultAlgoBBEE(TAcc const& acc,
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
                                                                   unsigned int fourthMDIndex) {
    bool isPS_InLo = (modules.moduleType()[innerInnerLowerModuleIndex] == PS);
    bool isPS_OutLo = (modules.moduleType()[outerInnerLowerModuleIndex] == PS);

    float rt_InLo = mds.anchorRt()[firstMDIndex];
    float rt_InOut = mds.anchorRt()[secondMDIndex];
    float rt_OutLo = mds.anchorRt()[thirdMDIndex];

    float z_InLo = mds.anchorZ()[firstMDIndex];
    float z_InOut = mds.anchorZ()[secondMDIndex];
    float z_OutLo = mds.anchorZ()[thirdMDIndex];

    float alpha1GeV_OutLo =
        alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * k2Rinv1GeVf / ptCut, kSinAlphaMax));

    float dzDrtScale =
        alpaka::math::tan(acc, alpha1GeV_OutLo) / alpha1GeV_OutLo;  // The track can bend in r-z plane slightly
    float zpitch_InLo = (isPS_InLo ? kPixelPSZpitch : kStrip2SZpitch);
    float zpitch_OutLo = (isPS_OutLo ? kPixelPSZpitch : kStrip2SZpitch);
    float zGeom = zpitch_InLo + zpitch_OutLo;

    // Cut #0: Preliminary (Only here in endcap case)
    if (z_InLo * z_OutLo <= 0)
      return false;

    float dLum = alpaka::math::copysign(acc, kDeltaZLum, z_InLo);
    bool isOutSgInnerMDPS = modules.moduleType()[outerInnerLowerModuleIndex] == PS;
    float rtGeom1 = isOutSgInnerMDPS ? kPixelPSZpitch : kStrip2SZpitch;
    float zGeom1 = alpaka::math::copysign(acc, zGeom, z_InLo);
    float rtLo = rt_InLo * (1.f + (z_OutLo - z_InLo - zGeom1) / (z_InLo + zGeom1 + dLum) / dzDrtScale) -
                 rtGeom1;  //slope correction only on the lower end
    float rtOut = rt_OutLo;

    //Cut #1: rt condition
    if (rtOut < rtLo)
      return false;

    float zInForHi = z_InLo - zGeom1 - dLum;
    if (zInForHi * z_InLo < 0) {
      zInForHi = alpaka::math::copysign(acc, 0.1f, z_InLo);
    }
    float rtHi = rt_InLo * (1.f + (z_OutLo - z_InLo + zGeom1) / zInForHi) + rtGeom1;

    //Cut #2: rt condition
    if ((rt_OutLo < rtLo) || (rt_OutLo > rtHi))
      return false;

    float rIn = alpaka::math::sqrt(acc, z_InLo * z_InLo + rt_InLo * rt_InLo);
    const float drtSDIn = rt_InOut - rt_InLo;
    const float dzSDIn = z_InOut - z_InLo;
    const float dr3SDIn = alpaka::math::sqrt(acc, rt_InOut * rt_InOut + z_InOut * z_InOut) -
                          alpaka::math::sqrt(acc, rt_InLo * rt_InLo + z_InLo * z_InLo);

    const float coshEta = dr3SDIn / drtSDIn;  //direction estimate
    const float dzOutInAbs = alpaka::math::abs(acc, z_OutLo - z_InLo);
    const float multDzDr = dzOutInAbs * coshEta / (coshEta * coshEta - 1.f);
    const float zGeom1_another = kPixelPSZpitch;
    float kZ = (z_OutLo - z_InLo) / dzSDIn;
    float drtErr =
        zGeom1_another * zGeom1_another * drtSDIn * drtSDIn / dzSDIn / dzSDIn * (1.f - 2.f * kZ + 2.f * kZ * kZ);
    const float thetaMuls2 = (kMulsInGeV * kMulsInGeV) * (0.1f + 0.2f * (rt_OutLo - rt_InLo) / 50.f) * (rIn / rt_InLo);
    const float muls2 = thetaMuls2 * 9.f / (ptCut * ptCut) * 16.f;
    drtErr += muls2 * multDzDr * multDzDr / 3.f * coshEta * coshEta;
    drtErr = alpaka::math::sqrt(acc, drtErr);

    //Cut #3: rt-z pointed
    if ((kZ < 0) || (rtOut < rtLo) || (rtOut > rtHi))
      return false;

    const float pvOffset = 0.1f / rt_OutLo;
    float dPhiCut = alpha1GeV_OutLo + alpaka::math::sqrt(acc, muls2 + pvOffset * pvOffset);

    float deltaPhiPos = phi_mpi_pi(acc, mds.anchorPhi()[fourthMDIndex] - mds.anchorPhi()[secondMDIndex]);

    //Cut #4: deltaPhiPos can be tighter
    if (alpaka::math::abs(acc, deltaPhiPos) > dPhiCut)
      return false;

    float midPointX = 0.5f * (mds.anchorX()[firstMDIndex] + mds.anchorX()[thirdMDIndex]);
    float midPointY = 0.5f * (mds.anchorY()[firstMDIndex] + mds.anchorY()[thirdMDIndex]);
    float diffX = mds.anchorX()[thirdMDIndex] - mds.anchorX()[firstMDIndex];
    float diffY = mds.anchorY()[thirdMDIndex] - mds.anchorY()[firstMDIndex];

    float dPhi = deltaPhi(acc, midPointX, midPointY, diffX, diffY);
    // Cut #5: deltaPhiChange
    if (alpaka::math::abs(acc, dPhi) > dPhiCut)
      return false;

    float sdIn_alpha = __H2F(segments.dPhiChanges()[innerSegmentIndex]);
    float sdIn_alpha_min = __H2F(segments.dPhiChangeMins()[innerSegmentIndex]);
    float sdIn_alpha_max = __H2F(segments.dPhiChangeMaxs()[innerSegmentIndex]);
    float sdOut_alpha = sdIn_alpha;

    float sdOut_dPhiPos = phi_mpi_pi(acc, mds.anchorPhi()[fourthMDIndex] - mds.anchorPhi()[thirdMDIndex]);

    float sdOut_dPhiChange = __H2F(segments.dPhiChanges()[outerSegmentIndex]);
    float sdOut_dPhiChange_min = __H2F(segments.dPhiChangeMins()[outerSegmentIndex]);
    float sdOut_dPhiChange_max = __H2F(segments.dPhiChangeMaxs()[outerSegmentIndex]);

    float sdOut_alphaOutRHmin = phi_mpi_pi(acc, sdOut_dPhiChange_min - sdOut_dPhiPos);
    float sdOut_alphaOutRHmax = phi_mpi_pi(acc, sdOut_dPhiChange_max - sdOut_dPhiPos);
    float sdOut_alphaOut = phi_mpi_pi(acc, sdOut_dPhiChange - sdOut_dPhiPos);

    float tl_axis_x = mds.anchorX()[fourthMDIndex] - mds.anchorX()[firstMDIndex];
    float tl_axis_y = mds.anchorY()[fourthMDIndex] - mds.anchorY()[firstMDIndex];

    float betaIn = sdIn_alpha - phi_mpi_pi(acc, phi(acc, tl_axis_x, tl_axis_y) - mds.anchorPhi()[firstMDIndex]);

    float betaInRHmin = betaIn;
    float betaInRHmax = betaIn;
    float betaOut = -sdOut_alphaOut + phi_mpi_pi(acc, phi(acc, tl_axis_x, tl_axis_y) - mds.anchorPhi()[fourthMDIndex]);

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
    const float corrF = 1.f;
    float betaInCut =
        alpaka::math::asin(acc, alpaka::math::min(acc, (-sdIn_dr * corrF + dr) * k2Rinv1GeVf / ptCut, kSinAlphaMax)) +
        (0.02f / sdIn_d);

    //Cut #6: first beta cut
    if (alpaka::math::abs(acc, betaInRHmin) >= betaInCut)
      return false;

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

    runDeltaBetaIterations(acc, betaIn, betaOut, betaAv, pt_beta, sdIn_dr, sdOut_dr, dr, lIn);

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
      dBetaROut = (alpaka::math::sqrt(acc,
                                      mds.anchorHighEdgeX()[fourthMDIndex] * mds.anchorHighEdgeX()[fourthMDIndex] +
                                          mds.anchorHighEdgeY()[fourthMDIndex] * mds.anchorHighEdgeY()[fourthMDIndex]) -
                   alpaka::math::sqrt(acc,
                                      mds.anchorLowEdgeX()[fourthMDIndex] * mds.anchorLowEdgeX()[fourthMDIndex] +
                                          mds.anchorLowEdgeY()[fourthMDIndex] * mds.anchorLowEdgeY()[fourthMDIndex])) *
                  sinDPhi / dr;
    }

    const float dBetaROut2 = dBetaROut * dBetaROut;
    float betaOutCut = alpaka::math::asin(acc, alpaka::math::min(acc, dr * k2Rinv1GeVf / ptCut, kSinAlphaMax)) +
                       (0.02f / sdOut_d) + alpaka::math::sqrt(acc, dBetaLum2 + dBetaMuls2);

    //Cut #6: The real beta cut
    if (alpaka::math::abs(acc, betaOut) >= betaOutCut)
      return false;

    float dBetaRes = 0.02f / alpaka::math::min(acc, sdOut_d, sdIn_d);
    float dBetaCut2 =
        (dBetaRes * dBetaRes * 2.0f + dBetaMuls2 + dBetaLum2 + dBetaRIn2 + dBetaROut2 +
         0.25f *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)) *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)));
    float dBeta = betaIn - betaOut;
    //Cut #7: Cut on dBet
    return dBeta * dBeta <= dBetaCut2;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuintupletDefaultAlgoEEEE(TAcc const& acc,
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
                                                                   unsigned int fourthMDIndex) {
    float rt_InLo = mds.anchorRt()[firstMDIndex];
    float rt_InOut = mds.anchorRt()[secondMDIndex];
    float rt_OutLo = mds.anchorRt()[thirdMDIndex];

    float z_InLo = mds.anchorZ()[firstMDIndex];
    float z_InOut = mds.anchorZ()[secondMDIndex];
    float z_OutLo = mds.anchorZ()[thirdMDIndex];

    float alpha1GeV_OutLo =
        alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * k2Rinv1GeVf / ptCut, kSinAlphaMax));

    float dzDrtScale =
        alpaka::math::tan(acc, alpha1GeV_OutLo) / alpha1GeV_OutLo;  // The track can bend in r-z plane slightly

    // Cut #0: Preliminary (Only here in endcap case)
    if ((z_InLo * z_OutLo) <= 0)
      return false;

    float dLum = alpaka::math::copysign(acc, kDeltaZLum, z_InLo);
    bool isOutSgInnerMDPS = modules.moduleType()[outerInnerLowerModuleIndex] == PS;
    bool isInSgInnerMDPS = modules.moduleType()[innerInnerLowerModuleIndex] == PS;

    float rtGeom = (isInSgInnerMDPS and isOutSgInnerMDPS)  ? 2.f * kPixelPSZpitch
                   : (isInSgInnerMDPS or isOutSgInnerMDPS) ? kPixelPSZpitch + kStrip2SZpitch
                                                           : 2.f * kStrip2SZpitch;

    float dz = z_OutLo - z_InLo;
    float rtLo = rt_InLo * (1.f + dz / (z_InLo + dLum) / dzDrtScale) - rtGeom;  //slope correction only on the lower end

    float rtOut = rt_OutLo;

    //Cut #1: rt condition

    float rtHi = rt_InLo * (1.f + dz / (z_InLo - dLum)) + rtGeom;

    if ((rtOut < rtLo) || (rtOut > rtHi))
      return false;

    bool isInSgOuterMDPS = modules.moduleType()[innerOuterLowerModuleIndex] == PS;

    const float drtSDIn = rt_InOut - rt_InLo;
    const float dzSDIn = z_InOut - z_InLo;
    const float dr3SDIn = alpaka::math::sqrt(acc, rt_InOut * rt_InOut + z_InOut * z_InOut) -
                          alpaka::math::sqrt(acc, rt_InLo * rt_InLo + z_InLo * z_InLo);
    float coshEta = dr3SDIn / drtSDIn;  //direction estimate
    float dzOutInAbs = alpaka::math::abs(acc, z_OutLo - z_InLo);
    float multDzDr = dzOutInAbs * coshEta / (coshEta * coshEta - 1.f);

    float kZ = (z_OutLo - z_InLo) / dzSDIn;
    float thetaMuls2 = (kMulsInGeV * kMulsInGeV) * (0.1f + 0.2f * (rt_OutLo - rt_InLo) / 50.f);

    float muls2 = thetaMuls2 * 9.f / (ptCut * ptCut) * 16.f;

    float drtErr =
        alpaka::math::sqrt(acc,
                           kPixelPSZpitch * kPixelPSZpitch * 2.f / (dzSDIn * dzSDIn) * (dzOutInAbs * dzOutInAbs) +
                               muls2 * multDzDr * multDzDr / 3.f * coshEta * coshEta);

    float drtMean = drtSDIn * dzOutInAbs / alpaka::math::abs(acc, dzSDIn);
    float rtWindow = drtErr + rtGeom;
    float rtLo_point = rt_InLo + drtMean / dzDrtScale - rtWindow;
    float rtHi_point = rt_InLo + drtMean + rtWindow;

    // Cut #3: rt-z pointed
    // https://github.com/slava77/cms-tkph2-ntuple/blob/superDoubletLinked-91X-noMock/doubletAnalysis.C#L3765

    if (isInSgInnerMDPS and isInSgOuterMDPS)  // If both PS then we can point
    {
      if (kZ < 0 || rtOut < rtLo_point || rtOut > rtHi_point)
        return false;
    }

    float pvOffset = 0.1f / rtOut;
    float dPhiCut = alpha1GeV_OutLo + alpaka::math::sqrt(acc, muls2 + pvOffset * pvOffset);

    float deltaPhiPos = phi_mpi_pi(acc, mds.anchorPhi()[fourthMDIndex] - mds.anchorPhi()[secondMDIndex]);

    if (alpaka::math::abs(acc, deltaPhiPos) > dPhiCut)
      return false;

    float midPointX = 0.5f * (mds.anchorX()[firstMDIndex] + mds.anchorX()[thirdMDIndex]);
    float midPointY = 0.5f * (mds.anchorY()[firstMDIndex] + mds.anchorY()[thirdMDIndex]);
    float diffX = mds.anchorX()[thirdMDIndex] - mds.anchorX()[firstMDIndex];
    float diffY = mds.anchorY()[thirdMDIndex] - mds.anchorY()[firstMDIndex];

    float dPhi = deltaPhi(acc, midPointX, midPointY, diffX, diffY);

    // Cut #5: deltaPhiChange
    if (alpaka::math::abs(acc, dPhi) > dPhiCut)
      return false;

    float sdIn_alpha = __H2F(segments.dPhiChanges()[innerSegmentIndex]);
    float sdOut_alpha = sdIn_alpha;  //weird
    float sdOut_dPhiPos = phi_mpi_pi(acc, mds.anchorPhi()[fourthMDIndex] - mds.anchorPhi()[thirdMDIndex]);

    float sdOut_dPhiChange = __H2F(segments.dPhiChanges()[outerSegmentIndex]);
    float sdOut_dPhiChange_min = __H2F(segments.dPhiChangeMins()[outerSegmentIndex]);
    float sdOut_dPhiChange_max = __H2F(segments.dPhiChangeMaxs()[outerSegmentIndex]);

    float sdOut_alphaOutRHmin = phi_mpi_pi(acc, sdOut_dPhiChange_min - sdOut_dPhiPos);
    float sdOut_alphaOutRHmax = phi_mpi_pi(acc, sdOut_dPhiChange_max - sdOut_dPhiPos);
    float sdOut_alphaOut = phi_mpi_pi(acc, sdOut_dPhiChange - sdOut_dPhiPos);

    float tl_axis_x = mds.anchorX()[fourthMDIndex] - mds.anchorX()[firstMDIndex];
    float tl_axis_y = mds.anchorY()[fourthMDIndex] - mds.anchorY()[firstMDIndex];

    float betaIn = sdIn_alpha - phi_mpi_pi(acc, phi(acc, tl_axis_x, tl_axis_y) - mds.anchorPhi()[firstMDIndex]);

    float sdIn_alphaRHmin = __H2F(segments.dPhiChangeMins()[innerSegmentIndex]);
    float sdIn_alphaRHmax = __H2F(segments.dPhiChangeMaxs()[innerSegmentIndex]);
    float betaInRHmin = betaIn + sdIn_alphaRHmin - sdIn_alpha;
    float betaInRHmax = betaIn + sdIn_alphaRHmax - sdIn_alpha;

    float betaOut = -sdOut_alphaOut + phi_mpi_pi(acc, phi(acc, tl_axis_x, tl_axis_y) - mds.anchorPhi()[fourthMDIndex]);

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
    const float corrF = 1.f;
    float betaInCut =
        alpaka::math::asin(acc, alpaka::math::min(acc, (-sdIn_dr * corrF + dr) * k2Rinv1GeVf / ptCut, kSinAlphaMax)) +
        (0.02f / sdIn_d);

    //Cut #6: first beta cut
    if (alpaka::math::abs(acc, betaInRHmin) >= betaInCut)
      return false;

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

    runDeltaBetaIterations(acc, betaIn, betaOut, betaAv, pt_beta, sdIn_dr, sdOut_dr, dr, lIn);

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

    const float dBetaRIn2 = 0;  // TODO-RH

    float dBetaROut2 = 0;  //TODO-RH
    float betaOutCut = alpaka::math::asin(acc, alpaka::math::min(acc, dr * k2Rinv1GeVf / ptCut, kSinAlphaMax)) +
                       (0.02f / sdOut_d) + alpaka::math::sqrt(acc, dBetaLum2 + dBetaMuls2);

    //Cut #6: The real beta cut
    if (alpaka::math::abs(acc, betaOut) >= betaOutCut)
      return false;

    float dBetaRes = 0.02f / alpaka::math::min(acc, sdOut_d, sdIn_d);
    float dBetaCut2 =
        (dBetaRes * dBetaRes * 2.0f + dBetaMuls2 + dBetaLum2 + dBetaRIn2 + dBetaROut2 +
         0.25f *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)) *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)));
    float dBeta = betaIn - betaOut;
    //Cut #7: Cut on dBeta
    return dBeta * dBeta <= dBetaCut2;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuintupletAlgoSelector(TAcc const& acc,
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
                                                                unsigned int fourthMDIndex) {
    short innerInnerLowerModuleSubdet = modules.subdets()[innerInnerLowerModuleIndex];
    short innerOuterLowerModuleSubdet = modules.subdets()[innerOuterLowerModuleIndex];
    short outerInnerLowerModuleSubdet = modules.subdets()[outerInnerLowerModuleIndex];
    short outerOuterLowerModuleSubdet = modules.subdets()[outerOuterLowerModuleIndex];

    if (innerInnerLowerModuleSubdet == Barrel and innerOuterLowerModuleSubdet == Barrel and
        outerInnerLowerModuleSubdet == Barrel and outerOuterLowerModuleSubdet == Barrel) {
      return runQuintupletDefaultAlgoBBBB(acc,
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
                                          fourthMDIndex);
    } else if (innerInnerLowerModuleSubdet == Barrel and innerOuterLowerModuleSubdet == Barrel and
               outerInnerLowerModuleSubdet == Endcap and outerOuterLowerModuleSubdet == Endcap) {
      return runQuintupletDefaultAlgoBBEE(acc,
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
                                          fourthMDIndex);
    } else if (innerInnerLowerModuleSubdet == Barrel and innerOuterLowerModuleSubdet == Barrel and
               outerInnerLowerModuleSubdet == Barrel and outerOuterLowerModuleSubdet == Endcap) {
      return runQuintupletDefaultAlgoBBBB(acc,
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
                                          fourthMDIndex);
    } else if (innerInnerLowerModuleSubdet == Barrel and innerOuterLowerModuleSubdet == Endcap and
               outerInnerLowerModuleSubdet == Endcap and outerOuterLowerModuleSubdet == Endcap) {
      return runQuintupletDefaultAlgoBBEE(acc,
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
                                          fourthMDIndex);
    } else if (innerInnerLowerModuleSubdet == Endcap and innerOuterLowerModuleSubdet == Endcap and
               outerInnerLowerModuleSubdet == Endcap and outerOuterLowerModuleSubdet == Endcap) {
      return runQuintupletDefaultAlgoEEEE(acc,
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
                                          fourthMDIndex);
    }

    return false;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuintupletDefaultAlgo(TAcc const& acc,
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
                                                               float& innerRadius,
                                                               float& outerRadius,
                                                               float& bridgeRadius,
                                                               float& regressionG,
                                                               float& regressionF,
                                                               float& regressionRadius,
                                                               float& rzChiSquared,
                                                               float& chiSquared,
                                                               float& nonAnchorChiSquared,
                                                               bool& tightCutFlag) {
    unsigned int firstSegmentIndex = triplets.segmentIndices()[innerTripletIndex][0];
    unsigned int secondSegmentIndex = triplets.segmentIndices()[innerTripletIndex][1];
    unsigned int thirdSegmentIndex = triplets.segmentIndices()[outerTripletIndex][0];
    unsigned int fourthSegmentIndex = triplets.segmentIndices()[outerTripletIndex][1];

    unsigned int innerOuterOuterMiniDoubletIndex =
        segments.mdIndices()[secondSegmentIndex][1];  //inner triplet outer segment outer MD index
    unsigned int outerInnerInnerMiniDoubletIndex =
        segments.mdIndices()[thirdSegmentIndex][0];  //outer triplet inner segment inner MD index

    //this cut reduces the number of candidates by a factor of 3, i.e., 2 out of 3 warps can end right here!
    if (innerOuterOuterMiniDoubletIndex != outerInnerInnerMiniDoubletIndex)
      return false;

    unsigned int firstMDIndex = segments.mdIndices()[firstSegmentIndex][0];
    unsigned int secondMDIndex = segments.mdIndices()[secondSegmentIndex][0];
    unsigned int thirdMDIndex = segments.mdIndices()[secondSegmentIndex][1];
    unsigned int fourthMDIndex = segments.mdIndices()[thirdSegmentIndex][1];
    unsigned int fifthMDIndex = segments.mdIndices()[fourthSegmentIndex][1];

    if (not runQuintupletAlgoSelector(acc,
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
                                      fourthMDIndex))
      return false;

    if (not runQuintupletAlgoSelector(acc,
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
                                      fifthMDIndex))
      return false;

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

    //construct the arrays
    float x1Vec[] = {x1, x1, x1};
    float y1Vec[] = {y1, y1, y1};
    float x2Vec[] = {x2, x2, x2};
    float y2Vec[] = {y2, y2, y2};
    float x3Vec[] = {x3, x3, x3};
    float y3Vec[] = {y3, y3, y3};

    if (modules.subdets()[lowerModuleIndex1] == Endcap and modules.moduleType()[lowerModuleIndex1] == TwoS) {
      x1Vec[1] = mds.anchorLowEdgeX()[firstMDIndex];
      x1Vec[2] = mds.anchorHighEdgeX()[firstMDIndex];

      y1Vec[1] = mds.anchorLowEdgeY()[firstMDIndex];
      y1Vec[2] = mds.anchorHighEdgeY()[firstMDIndex];
    }
    if (modules.subdets()[lowerModuleIndex2] == Endcap and modules.moduleType()[lowerModuleIndex2] == TwoS) {
      x2Vec[1] = mds.anchorLowEdgeX()[secondMDIndex];
      x2Vec[2] = mds.anchorHighEdgeX()[secondMDIndex];

      y2Vec[1] = mds.anchorLowEdgeY()[secondMDIndex];
      y2Vec[2] = mds.anchorHighEdgeY()[secondMDIndex];
    }
    if (modules.subdets()[lowerModuleIndex3] == Endcap and modules.moduleType()[lowerModuleIndex3] == TwoS) {
      x3Vec[1] = mds.anchorLowEdgeX()[thirdMDIndex];
      x3Vec[2] = mds.anchorHighEdgeX()[thirdMDIndex];

      y3Vec[1] = mds.anchorLowEdgeY()[thirdMDIndex];
      y3Vec[2] = mds.anchorHighEdgeY()[thirdMDIndex];
    }

    float innerRadiusMin2S, innerRadiusMax2S;
    computeErrorInRadius(acc, x1Vec, y1Vec, x2Vec, y2Vec, x3Vec, y3Vec, innerRadiusMin2S, innerRadiusMax2S);

    for (int i = 0; i < 3; i++) {
      x1Vec[i] = x4;
      y1Vec[i] = y4;
    }
    if (modules.subdets()[lowerModuleIndex4] == Endcap and modules.moduleType()[lowerModuleIndex4] == TwoS) {
      x1Vec[1] = mds.anchorLowEdgeX()[fourthMDIndex];
      x1Vec[2] = mds.anchorHighEdgeX()[fourthMDIndex];

      y1Vec[1] = mds.anchorLowEdgeY()[fourthMDIndex];
      y1Vec[2] = mds.anchorHighEdgeY()[fourthMDIndex];
    }

    float bridgeRadiusMin2S, bridgeRadiusMax2S;
    computeErrorInRadius(acc, x2Vec, y2Vec, x3Vec, y3Vec, x1Vec, y1Vec, bridgeRadiusMin2S, bridgeRadiusMax2S);

    for (int i = 0; i < 3; i++) {
      x2Vec[i] = x5;
      y2Vec[i] = y5;
    }
    if (modules.subdets()[lowerModuleIndex5] == Endcap and modules.moduleType()[lowerModuleIndex5] == TwoS) {
      x2Vec[1] = mds.anchorLowEdgeX()[fifthMDIndex];
      x2Vec[2] = mds.anchorHighEdgeX()[fifthMDIndex];

      y2Vec[1] = mds.anchorLowEdgeY()[fifthMDIndex];
      y2Vec[2] = mds.anchorHighEdgeY()[fifthMDIndex];
    }

    float outerRadiusMin2S, outerRadiusMax2S;
    computeErrorInRadius(acc, x3Vec, y3Vec, x1Vec, y1Vec, x2Vec, y2Vec, outerRadiusMin2S, outerRadiusMax2S);

    float g, f;
    outerRadius = triplets.radius()[outerTripletIndex];
    bridgeRadius = computeRadiusFromThreeAnchorHits(acc, x2, y2, x3, y3, x4, y4, g, f);
    innerRadius = triplets.radius()[innerTripletIndex];
    g = triplets.centerX()[innerTripletIndex];
    f = triplets.centerY()[innerTripletIndex];

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
                               f,
                               tightCutFlag))
      return false;

    if (innerRadius < 0.95f * ptCut / (2.f * k2Rinv1GeVf))
      return false;

    //split by category
    bool matchedRadii;
    if (modules.subdets()[lowerModuleIndex1] == Barrel and modules.subdets()[lowerModuleIndex2] == Barrel and
        modules.subdets()[lowerModuleIndex3] == Barrel and modules.subdets()[lowerModuleIndex4] == Barrel and
        modules.subdets()[lowerModuleIndex5] == Barrel) {
      matchedRadii = matchRadiiBBBBB(acc, innerRadius, bridgeRadius, outerRadius);
    } else if (modules.subdets()[lowerModuleIndex1] == Barrel and modules.subdets()[lowerModuleIndex2] == Barrel and
               modules.subdets()[lowerModuleIndex3] == Barrel and modules.subdets()[lowerModuleIndex4] == Barrel and
               modules.subdets()[lowerModuleIndex5] == Endcap) {
      matchedRadii = matchRadiiBBBBE(acc, innerRadius, bridgeRadius, outerRadius);
    } else if (modules.subdets()[lowerModuleIndex1] == Barrel and modules.subdets()[lowerModuleIndex2] == Barrel and
               modules.subdets()[lowerModuleIndex3] == Barrel and modules.subdets()[lowerModuleIndex4] == Endcap and
               modules.subdets()[lowerModuleIndex5] == Endcap) {
      if (modules.layers()[lowerModuleIndex1] == 1) {
        matchedRadii =
            matchRadiiBBBEE12378(acc, innerRadius, bridgeRadius, outerRadius, bridgeRadiusMin2S, bridgeRadiusMax2S);
      } else if (modules.layers()[lowerModuleIndex1] == 2) {
        matchedRadii =
            matchRadiiBBBEE23478(acc, innerRadius, bridgeRadius, outerRadius, bridgeRadiusMin2S, bridgeRadiusMax2S);
      } else {
        matchedRadii =
            matchRadiiBBBEE34578(acc, innerRadius, bridgeRadius, outerRadius, bridgeRadiusMin2S, bridgeRadiusMax2S);
      }
    }

    else if (modules.subdets()[lowerModuleIndex1] == Barrel and modules.subdets()[lowerModuleIndex2] == Barrel and
             modules.subdets()[lowerModuleIndex3] == Endcap and modules.subdets()[lowerModuleIndex4] == Endcap and
             modules.subdets()[lowerModuleIndex5] == Endcap) {
      matchedRadii = matchRadiiBBEEE(acc, innerRadius, bridgeRadius, outerRadius, bridgeRadiusMin2S, bridgeRadiusMax2S);
    } else if (modules.subdets()[lowerModuleIndex1] == Barrel and modules.subdets()[lowerModuleIndex2] == Endcap and
               modules.subdets()[lowerModuleIndex3] == Endcap and modules.subdets()[lowerModuleIndex4] == Endcap and
               modules.subdets()[lowerModuleIndex5] == Endcap) {
      matchedRadii = matchRadiiBEEEE(acc,
                                     innerRadius,
                                     bridgeRadius,
                                     outerRadius,
                                     innerRadiusMin2S,
                                     innerRadiusMax2S,
                                     bridgeRadiusMin2S,
                                     bridgeRadiusMax2S);
    } else {
      matchedRadii = matchRadiiEEEEE(acc,
                                     innerRadius,
                                     bridgeRadius,
                                     outerRadius,
                                     innerRadiusMin2S,
                                     innerRadiusMax2S,
                                     bridgeRadiusMin2S,
                                     bridgeRadiusMax2S);
    }

    //compute regression radius right here - this computation is expensive!!!
    if (not matchedRadii)
      return false;

    float xVec[] = {x1, x2, x3, x4, x5};
    float yVec[] = {y1, y2, y3, y4, y5};
    const uint16_t lowerModuleIndices[] = {
        lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5};

    // 5 categories for sigmas
    float sigmas2[5], delta1[5], delta2[5], slopes[5];
    bool isFlat[5];

    computeSigmasForRegression(acc, modules, lowerModuleIndices, delta1, delta2, slopes, isFlat);
    regressionRadius = computeRadiusUsingRegression(acc,
                                                    Params_T5::kLayers,
                                                    xVec,
                                                    yVec,
                                                    delta1,
                                                    delta2,
                                                    slopes,
                                                    isFlat,
                                                    regressionG,
                                                    regressionF,
                                                    sigmas2,
                                                    chiSquared);

    unsigned int mdIndices[] = {firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, fifthMDIndex};
    float inference = t5dnn::runInference(acc,
                                          modules,
                                          mds,
                                          segments,
                                          triplets,
                                          xVec,
                                          yVec,
                                          mdIndices,
                                          lowerModuleIndices,
                                          innerTripletIndex,
                                          outerTripletIndex,
                                          innerRadius,
                                          outerRadius,
                                          bridgeRadius);
    tightCutFlag = tightCutFlag and (inference > t5dnn::kLSTWp2);  // T5-in-TC cut
    if (inference <= t5dnn::kLSTWp2)                               // T5-building cut
      return false;

    //compute the other chisquared
    //non anchor is always shifted for tilted and endcap!
    float nonAnchorDelta1[Params_T5::kLayers], nonAnchorDelta2[Params_T5::kLayers], nonAnchorSlopes[Params_T5::kLayers];
    float nonAnchorxs[] = {mds.outerX()[firstMDIndex],
                           mds.outerX()[secondMDIndex],
                           mds.outerX()[thirdMDIndex],
                           mds.outerX()[fourthMDIndex],
                           mds.outerX()[fifthMDIndex]};
    float nonAnchorys[] = {mds.outerY()[firstMDIndex],
                           mds.outerY()[secondMDIndex],
                           mds.outerY()[thirdMDIndex],
                           mds.outerY()[fourthMDIndex],
                           mds.outerY()[fifthMDIndex]};

    computeSigmasForRegression(acc,
                               modules,
                               lowerModuleIndices,
                               nonAnchorDelta1,
                               nonAnchorDelta2,
                               nonAnchorSlopes,
                               isFlat,
                               Params_T5::kLayers,
                               false);
    nonAnchorChiSquared = computeChiSquared(acc,
                                            Params_T5::kLayers,
                                            nonAnchorxs,
                                            nonAnchorys,
                                            nonAnchorDelta1,
                                            nonAnchorDelta2,
                                            nonAnchorSlopes,
                                            isFlat,
                                            regressionG,
                                            regressionF,
                                            regressionRadius);
    return true;
  }

  struct CreateQuintuplets {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  ModulesConst modules,
                                  MiniDoubletsConst mds,
                                  SegmentsConst segments,
                                  Triplets triplets,
                                  TripletsOccupancyConst tripletsOccupancy,
                                  Quintuplets quintuplets,
                                  QuintupletsOccupancy quintupletsOccupancy,
                                  ObjectRangesConst ranges,
                                  uint16_t nEligibleT5Modules) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (int iter = globalThreadIdx[0]; iter < nEligibleT5Modules; iter += gridThreadExtent[0]) {
        uint16_t lowerModule1 = ranges.indicesOfEligibleT5Modules()[iter];
        short layer2_adjustment;
        int layer = modules.layers()[lowerModule1];
        if (layer == 1) {
          layer2_adjustment = 1;
        }  // get upper segment to be in second layer
        else if (layer == 2) {
          layer2_adjustment = 0;
        }  // get lower segment to be in second layer
        else {
          continue;
        }
        unsigned int nInnerTriplets = tripletsOccupancy.nTriplets()[lowerModule1];
        for (unsigned int innerTripletArrayIndex = globalThreadIdx[1]; innerTripletArrayIndex < nInnerTriplets;
             innerTripletArrayIndex += gridThreadExtent[1]) {
          unsigned int innerTripletIndex = ranges.tripletModuleIndices()[lowerModule1] + innerTripletArrayIndex;
          uint16_t lowerModule2 = triplets.lowerModuleIndices()[innerTripletIndex][1];
          uint16_t lowerModule3 = triplets.lowerModuleIndices()[innerTripletIndex][2];
          unsigned int nOuterTriplets = tripletsOccupancy.nTriplets()[lowerModule3];
          for (unsigned int outerTripletArrayIndex = globalThreadIdx[2]; outerTripletArrayIndex < nOuterTriplets;
               outerTripletArrayIndex += gridThreadExtent[2]) {
            unsigned int outerTripletIndex = ranges.tripletModuleIndices()[lowerModule3] + outerTripletArrayIndex;
            uint16_t lowerModule4 = triplets.lowerModuleIndices()[outerTripletIndex][1];
            uint16_t lowerModule5 = triplets.lowerModuleIndices()[outerTripletIndex][2];

            float innerRadius, outerRadius, bridgeRadius, regressionG, regressionF, regressionRadius, rzChiSquared,
                chiSquared, nonAnchorChiSquared;  //required for making distributions

            bool tightCutFlag = false;
            bool success = runQuintupletDefaultAlgo(acc,
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
                                                    regressionG,
                                                    regressionF,
                                                    regressionRadius,
                                                    rzChiSquared,
                                                    chiSquared,
                                                    nonAnchorChiSquared,
                                                    tightCutFlag);

            if (success) {
              int totOccupancyQuintuplets = alpaka::atomicAdd(
                  acc, &quintupletsOccupancy.totOccupancyQuintuplets()[lowerModule1], 1u, alpaka::hierarchy::Threads{});
              if (totOccupancyQuintuplets >= ranges.quintupletModuleOccupancy()[lowerModule1]) {
#ifdef WARNINGS
                printf("Quintuplet excess alert! Module index = %d\n", lowerModule1);
#endif
              } else {
                int quintupletModuleIndex = alpaka::atomicAdd(
                    acc, &quintupletsOccupancy.nQuintuplets()[lowerModule1], 1u, alpaka::hierarchy::Threads{});
                //this if statement should never get executed!
                if (ranges.quintupletModuleIndices()[lowerModule1] == -1) {
#ifdef WARNINGS
                  printf("Quintuplets : no memory for module at module index = %d\n", lowerModule1);
#endif
                } else {
                  unsigned int quintupletIndex = ranges.quintupletModuleIndices()[lowerModule1] + quintupletModuleIndex;
                  float phi = mds.anchorPhi()[segments.mdIndices()[triplets.segmentIndices()[innerTripletIndex][0]]
                                                                  [layer2_adjustment]];
                  float eta = mds.anchorEta()[segments.mdIndices()[triplets.segmentIndices()[innerTripletIndex][0]]
                                                                  [layer2_adjustment]];
                  float pt = (innerRadius + outerRadius) * k2Rinv1GeVf;
                  float scores = chiSquared + nonAnchorChiSquared;
                  addQuintupletToMemory(triplets,
                                        quintuplets,
                                        innerTripletIndex,
                                        outerTripletIndex,
                                        lowerModule1,
                                        lowerModule2,
                                        lowerModule3,
                                        lowerModule4,
                                        lowerModule5,
                                        innerRadius,
                                        bridgeRadius,
                                        outerRadius,
                                        regressionG,
                                        regressionF,
                                        regressionRadius,
                                        rzChiSquared,
                                        chiSquared,
                                        nonAnchorChiSquared,
                                        pt,
                                        eta,
                                        phi,
                                        scores,
                                        layer,
                                        quintupletIndex,
                                        tightCutFlag);

                  triplets.partOfT5()[quintuplets.tripletIndices()[quintupletIndex][0]] = true;
                  triplets.partOfT5()[quintuplets.tripletIndices()[quintupletIndex][1]] = true;
                }
              }
            }
          }
        }
      }
    }
  };

  struct CreateEligibleModulesListForQuintuplets {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  ModulesConst modules,
                                  TripletsOccupancyConst tripletsOccupancy,
                                  ObjectRanges ranges) const {
      // implementation is 1D with a single block
      static_assert(std::is_same_v<TAcc, ALPAKA_ACCELERATOR_NAMESPACE::Acc1D>, "Should be Acc1D");
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));

      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      // Initialize variables in shared memory and set to 0
      int& nEligibleT5Modulesx = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      int& nTotalQuintupletsx = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      if (cms::alpakatools::once_per_block(acc)) {
        nTotalQuintupletsx = 0;
        nEligibleT5Modulesx = 0;
      }
      alpaka::syncBlockThreads(acc);

      for (int i = globalThreadIdx[0]; i < modules.nLowerModules(); i += gridThreadExtent[0]) {
        // Condition for a quintuple to exist for a module
        // TCs don't exist for layers 5 and 6 barrel, and layers 2,3,4,5 endcap
        short module_rings = modules.rings()[i];
        short module_layers = modules.layers()[i];
        short module_subdets = modules.subdets()[i];
        float module_eta = alpaka::math::abs(acc, modules.eta()[i]);

        if (tripletsOccupancy.nTriplets()[i] == 0)
          continue;
        if (module_subdets == Barrel and module_layers >= 3)
          continue;
        if (module_subdets == Endcap and module_layers > 1)
          continue;

        int nEligibleT5Modules = alpaka::atomicAdd(acc, &nEligibleT5Modulesx, 1, alpaka::hierarchy::Threads{});

        int category_number;
        if (module_layers <= 3 && module_subdets == 5)
          category_number = 0;
        else if (module_layers >= 4 && module_subdets == 5)
          category_number = 1;
        else if (module_layers <= 2 && module_subdets == 4 && module_rings >= 11)
          category_number = 2;
        else if (module_layers >= 3 && module_subdets == 4 && module_rings >= 8)
          category_number = 2;
        else if (module_layers <= 2 && module_subdets == 4 && module_rings <= 10)
          category_number = 3;
        else if (module_layers >= 3 && module_subdets == 4 && module_rings <= 7)
          category_number = 3;
        else
          category_number = -1;

        int eta_number;
        if (module_eta < 0.75f)
          eta_number = 0;
        else if (module_eta < 1.5f)
          eta_number = 1;
        else if (module_eta < 2.25f)
          eta_number = 2;
        else if (module_eta < 3.0f)
          eta_number = 3;
        else
          eta_number = -1;

        int occupancy;
        if (category_number == 0 && eta_number == 0)
          occupancy = 336;
        else if (category_number == 0 && eta_number == 1)
          occupancy = 414;
        else if (category_number == 0 && eta_number == 2)
          occupancy = 231;
        else if (category_number == 0 && eta_number == 3)
          occupancy = 146;
        else if (category_number == 3 && eta_number == 1)
          occupancy = 0;
        else if (category_number == 3 && eta_number == 2)
          occupancy = 191;
        else if (category_number == 3 && eta_number == 3)
          occupancy = 106;
        else {
          occupancy = 0;
#ifdef WARNINGS
          printf("Unhandled case in createEligibleModulesListForQuintupletsGPU! Module index = %i\n", i);
#endif
        }

        int nTotQ = alpaka::atomicAdd(acc, &nTotalQuintupletsx, occupancy, alpaka::hierarchy::Threads{});
        ranges.quintupletModuleIndices()[i] = nTotQ;
        ranges.indicesOfEligibleT5Modules()[nEligibleT5Modules] = i;
        ranges.quintupletModuleOccupancy()[i] = occupancy;
      }

      // Wait for all threads to finish before reporting final values
      alpaka::syncBlockThreads(acc);
      if (cms::alpakatools::once_per_block(acc)) {
        ranges.nEligibleT5Modules() = static_cast<uint16_t>(nEligibleT5Modulesx);
        ranges.nTotalQuints() = static_cast<unsigned int>(nTotalQuintupletsx);
      }
    }
  };

  struct AddQuintupletRangesToEventExplicit {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  ModulesConst modules,
                                  QuintupletsOccupancyConst quintupletsOccupancy,
                                  ObjectRanges ranges) const {
      // implementation is 1D with a single block
      static_assert(std::is_same_v<TAcc, ALPAKA_ACCELERATOR_NAMESPACE::Acc1D>, "Should be Acc1D");
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));

      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (uint16_t i = globalThreadIdx[0]; i < modules.nLowerModules(); i += gridThreadExtent[0]) {
        if (quintupletsOccupancy.nQuintuplets()[i] == 0 or ranges.quintupletModuleIndices()[i] == -1) {
          ranges.quintupletRanges()[i][0] = -1;
          ranges.quintupletRanges()[i][1] = -1;
        } else {
          ranges.quintupletRanges()[i][0] = ranges.quintupletModuleIndices()[i];
          ranges.quintupletRanges()[i][1] =
              ranges.quintupletModuleIndices()[i] + quintupletsOccupancy.nQuintuplets()[i] - 1;
        }
      }
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
