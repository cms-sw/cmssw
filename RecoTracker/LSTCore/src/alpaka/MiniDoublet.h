#ifndef RecoTracker_LSTCore_src_alpaka_MiniDoublet_h
#define RecoTracker_LSTCore_src_alpaka_MiniDoublet_h

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/MiniDoubletsSoA.h"
#include "RecoTracker/LSTCore/interface/alpaka/MiniDoubletsDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/ModulesSoA.h"
#include "RecoTracker/LSTCore/interface/EndcapGeometry.h"
#include "RecoTracker/LSTCore/interface/ObjectRangesSoA.h"

#include "Hit.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addMDToMemory(TAcc const& acc,
                                                    MiniDoublets mds,
                                                    HitsConst hits,
                                                    ModulesConst modules,
                                                    unsigned int lowerHitIdx,
                                                    unsigned int upperHitIdx,
                                                    uint16_t lowerModuleIdx,
                                                    float dz,
                                                    float dPhi,
                                                    float dPhiChange,
                                                    float shiftedX,
                                                    float shiftedY,
                                                    float shiftedZ,
                                                    float noShiftedDphi,
                                                    float noShiftedDPhiChange,
                                                    unsigned int idx) {
    //the index into which this MD needs to be written will be computed in the kernel
    //nMDs variable will be incremented in the kernel, no need to worry about that here

    mds.moduleIndices()[idx] = lowerModuleIdx;
    unsigned int anchorHitIndex, outerHitIndex;
    if (modules.moduleType()[lowerModuleIdx] == PS and modules.moduleLayerType()[lowerModuleIdx] == Strip) {
      mds.anchorHitIndices()[idx] = upperHitIdx;
      mds.outerHitIndices()[idx] = lowerHitIdx;

      anchorHitIndex = upperHitIdx;
      outerHitIndex = lowerHitIdx;
    } else {
      mds.anchorHitIndices()[idx] = lowerHitIdx;
      mds.outerHitIndices()[idx] = upperHitIdx;

      anchorHitIndex = lowerHitIdx;
      outerHitIndex = upperHitIdx;
    }

    mds.dphichanges()[idx] = dPhiChange;

    mds.dphis()[idx] = dPhi;
    mds.dzs()[idx] = dz;
    mds.shiftedXs()[idx] = shiftedX;
    mds.shiftedYs()[idx] = shiftedY;
    mds.shiftedZs()[idx] = shiftedZ;

    mds.noShiftedDphis()[idx] = noShiftedDphi;
    mds.noShiftedDphiChanges()[idx] = noShiftedDPhiChange;

    mds.anchorX()[idx] = hits.xs()[anchorHitIndex];
    mds.anchorY()[idx] = hits.ys()[anchorHitIndex];
    mds.anchorZ()[idx] = hits.zs()[anchorHitIndex];
    mds.anchorRt()[idx] = hits.rts()[anchorHitIndex];
    mds.anchorPhi()[idx] = hits.phis()[anchorHitIndex];
    mds.anchorEta()[idx] = hits.etas()[anchorHitIndex];
    mds.anchorHighEdgeX()[idx] = hits.highEdgeXs()[anchorHitIndex];
    mds.anchorHighEdgeY()[idx] = hits.highEdgeYs()[anchorHitIndex];
    mds.anchorLowEdgeX()[idx] = hits.lowEdgeXs()[anchorHitIndex];
    mds.anchorLowEdgeY()[idx] = hits.lowEdgeYs()[anchorHitIndex];
    mds.anchorHighEdgePhi()[idx] = alpaka::math::atan2(acc, mds.anchorHighEdgeY()[idx], mds.anchorHighEdgeX()[idx]);
    mds.anchorLowEdgePhi()[idx] = alpaka::math::atan2(acc, mds.anchorLowEdgeY()[idx], mds.anchorLowEdgeX()[idx]);

    mds.outerX()[idx] = hits.xs()[outerHitIndex];
    mds.outerY()[idx] = hits.ys()[outerHitIndex];
    mds.outerZ()[idx] = hits.zs()[outerHitIndex];
    mds.outerRt()[idx] = hits.rts()[outerHitIndex];
    mds.outerPhi()[idx] = hits.phis()[outerHitIndex];
    mds.outerEta()[idx] = hits.etas()[outerHitIndex];
    mds.outerHighEdgeX()[idx] = hits.highEdgeXs()[outerHitIndex];
    mds.outerHighEdgeY()[idx] = hits.highEdgeYs()[outerHitIndex];
    mds.outerLowEdgeX()[idx] = hits.lowEdgeXs()[outerHitIndex];
    mds.outerLowEdgeY()[idx] = hits.lowEdgeYs()[outerHitIndex];
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool isTighterTiltedModules(ModulesConst modules, uint16_t moduleIndex) {
    // The "tighter" tilted modules are the subset of tilted modules that have smaller spacing
    // This is the same as what was previously considered as"isNormalTiltedModules"
    // See Figure 9.1 of https://cds.cern.ch/record/2272264/files/CMS-TDR-014.pdf
    short subdet = modules.subdets()[moduleIndex];
    short layer = modules.layers()[moduleIndex];
    short side = modules.sides()[moduleIndex];
    short rod = modules.rods()[moduleIndex];

    if (subdet == Barrel) {
      if ((side != Center and layer == 3) or (side == NegZ and layer == 2 and rod > 5) or
          (side == PosZ and layer == 2 and rod < 8) or (side == NegZ and layer == 1 and rod > 9) or
          (side == PosZ and layer == 1 and rod < 4))
        return true;
      else
        return false;
    } else
      return false;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float moduleGapSize(ModulesConst modules, uint16_t moduleIndex) {
    unsigned int iL = modules.layers()[moduleIndex] - 1;
    unsigned int iR = modules.rings()[moduleIndex] - 1;
    short subdet = modules.subdets()[moduleIndex];
    short side = modules.sides()[moduleIndex];

    float moduleSeparation = 0;

    if (subdet == Barrel and side == Center) {
      moduleSeparation = kMiniDeltaFlat[iL];
    } else if (isTighterTiltedModules(modules, moduleIndex)) {
      moduleSeparation = kMiniDeltaTilted[iL];
    } else if (subdet == Endcap) {
      moduleSeparation = kMiniDeltaEndcap[iL][iR];
    } else  //Loose tilted modules
    {
      moduleSeparation = kMiniDeltaLooseTilted[iL];
    }

    return moduleSeparation;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float dPhiThreshold(
      TAcc const& acc, float rt, ModulesConst modules, uint16_t moduleIndex, float dPhi = 0, float dz = 0) {
    // =================================================================
    // Various constants
    // =================================================================
    //mean of the horizontal layer position in y; treat this as R below

    // =================================================================
    // Computing some components that make up the cut threshold
    // =================================================================

    unsigned int iL = modules.layers()[moduleIndex] - 1;
    const float miniSlope = alpaka::math::asin(acc, alpaka::math::min(acc, rt * k2Rinv1GeVf / ptCut, kSinAlphaMax));
    const float rLayNominal =
        ((modules.subdets()[moduleIndex] == Barrel) ? kMiniRminMeanBarrel[iL] : kMiniRminMeanEndcap[iL]);
    const float miniPVoff = 0.1f / rLayNominal;
    const float miniMuls = ((modules.subdets()[moduleIndex] == Barrel) ? kMiniMulsPtScaleBarrel[iL] * 3.f / ptCut
                                                                       : kMiniMulsPtScaleEndcap[iL] * 3.f / ptCut);
    const bool isTilted = modules.subdets()[moduleIndex] == Barrel and modules.sides()[moduleIndex] != Center;
    //the lower module is sent in irrespective of its layer type. We need to fetch the drdz properly

    float drdz;
    if (isTilted) {
      if (modules.moduleType()[moduleIndex] == PS and modules.moduleLayerType()[moduleIndex] == Strip) {
        drdz = modules.drdzs()[moduleIndex];
      } else {
        drdz = modules.drdzs()[modules.partnerModuleIndices()[moduleIndex]];
      }
    } else {
      drdz = 0;
    }
    const float miniTilt2 = ((isTilted) ? (0.5f * 0.5f) * (kPixelPSZpitch * kPixelPSZpitch) * (drdz * drdz) /
                                              (1.f + drdz * drdz) / moduleGapSize(modules, moduleIndex)
                                        : 0);

    // Compute luminous region requirement for endcap
    const float miniLum = alpaka::math::abs(acc, dPhi * kDeltaZLum / dz);  // Balaji's new error

    // =================================================================
    // Return the threshold value
    // =================================================================
    // Following condition is met if the module is central and flatly lying
    if (modules.subdets()[moduleIndex] == Barrel and modules.sides()[moduleIndex] == Center) {
      return miniSlope + alpaka::math::sqrt(acc, miniMuls * miniMuls + miniPVoff * miniPVoff);
    }
    // Following condition is met if the module is central and tilted
    else if (modules.subdets()[moduleIndex] == Barrel and
             modules.sides()[moduleIndex] != Center)  //all types of tilted modules
    {
      return miniSlope +
             alpaka::math::sqrt(acc, miniMuls * miniMuls + miniPVoff * miniPVoff + miniTilt2 * miniSlope * miniSlope);
    }
    // If not barrel, it is Endcap
    else {
      return miniSlope + alpaka::math::sqrt(acc, miniMuls * miniMuls + miniPVoff * miniPVoff + miniLum * miniLum);
    }
  }

  template <typename TAcc>
  ALPAKA_FN_INLINE ALPAKA_FN_ACC void shiftStripHits(TAcc const& acc,
                                                     ModulesConst modules,
                                                     uint16_t lowerModuleIndex,
                                                     uint16_t upperModuleIndex,
                                                     unsigned int lowerHitIndex,
                                                     unsigned int upperHitIndex,
                                                     float* shiftedCoords,
                                                     float xLower,
                                                     float yLower,
                                                     float zLower,
                                                     float rtLower,
                                                     float xUpper,
                                                     float yUpper,
                                                     float zUpper,
                                                     float rtUpper) {
    // This is the strip shift scheme that is explained in http://uaf-10.t2.ucsd.edu/~phchang/talks/PhilipChang20190607_SDL_Update.pdf (see backup slides)
    // The main feature of this shifting is that the strip hits are shifted to be "aligned" in the line of sight from interaction point to the the pixel hit.
    // (since pixel hit is well defined in 3-d)
    // The strip hit is shifted along the strip detector to be placed in a guessed position where we think they would have actually crossed
    // The size of the radial direction shift due to module separation gap is computed in "radial" size, while the shift is done along the actual strip orientation
    // This means that there may be very very subtle edge effects coming from whether the strip hit is center of the module or the at the edge of the module
    // But this should be relatively minor effect

    // dependent variables for this if statement
    // lowerModule
    // lowerHit
    // upperHit
    // endcapGeometry
    // tiltedGeometry

    // Some variables relevant to the function
    float xp;       // pixel x (pixel hit x)
    float yp;       // pixel y (pixel hit y)
    float zp;       // pixel y (pixel hit y)
    float rtp;      // pixel y (pixel hit y)
    float xa;       // "anchor" x (the anchor position on the strip module plane from pixel hit)
    float ya;       // "anchor" y (the anchor position on the strip module plane from pixel hit)
    float xo;       // old x (before the strip hit is moved up or down)
    float yo;       // old y (before the strip hit is moved up or down)
    float xn;       // new x (after the strip hit is moved up or down)
    float yn;       // new y (after the strip hit is moved up or down)
    float abszn;    // new z in absolute value
    float zn;       // new z with the sign (+/-) accounted
    float angleA;   // in r-z plane the theta of the pixel hit in polar coordinate is the angleA
    float angleB;   // this is the angle of tilted module in r-z plane ("drdz"), for endcap this is 90 degrees
    bool isEndcap;  // If endcap, drdz = infinity
    float moduleSeparation;
    float drprime;    // The radial shift size in x-y plane projection
    float drprime_x;  // x-component of drprime
    float drprime_y;  // y-component of drprime
    const float& slope =
        modules.dxdys()[lowerModuleIndex];  // The slope of the possible strip hits for a given module in x-y plane
    float absArctanSlope;
    float angleM;  // the angle M is the angle of rotation of the module in x-y plane if the possible strip hits are along the x-axis, then angleM = 0, and if the possible strip hits are along y-axis angleM = 90 degrees
    float absdzprime;  // The distance between the two points after shifting
    const float& drdz_ = modules.drdzs()[lowerModuleIndex];
    // Assign hit pointers based on their hit type
    if (modules.moduleType()[lowerModuleIndex] == PS) {
      // TODO: This is somewhat of an mystery.... somewhat confused why this is the case
      if (modules.subdets()[lowerModuleIndex] == Barrel ? modules.moduleLayerType()[lowerModuleIndex] != Pixel
                                                        : modules.moduleLayerType()[lowerModuleIndex] == Pixel) {
        xo = xUpper;
        yo = yUpper;
        xp = xLower;
        yp = yLower;
        zp = zLower;
        rtp = rtLower;
      } else {
        xo = xLower;
        yo = yLower;
        xp = xUpper;
        yp = yUpper;
        zp = zUpper;
        rtp = rtUpper;
      }
    } else {
      xo = xUpper;
      yo = yUpper;
      xp = xLower;
      yp = yLower;
      zp = zLower;
      rtp = rtLower;
    }

    // If it is endcap some of the math gets simplified (and also computers don't like infinities)
    isEndcap = modules.subdets()[lowerModuleIndex] == Endcap;

    // NOTE: TODO: Keep in mind that the sin(atan) function can be simplified to something like x / sqrt(1 + x^2) and similar for cos
    // I am not sure how slow sin, atan, cos, functions are in c++. If x / sqrt(1 + x^2) are faster change this later to reduce arithmetic computation time
    angleA = alpaka::math::abs(acc, alpaka::math::atan(acc, rtp / zp));
    angleB =
        ((isEndcap)
             ? kPi / 2.f
             : alpaka::math::atan(
                   acc,
                   drdz_));  // The tilt module on the positive z-axis has negative drdz slope in r-z plane and vice versa

    moduleSeparation = moduleGapSize(modules, lowerModuleIndex);

    // Sign flips if the pixel is later layer
    if (modules.moduleType()[lowerModuleIndex] == PS and modules.moduleLayerType()[lowerModuleIndex] != Pixel) {
      moduleSeparation *= -1;
    }

    drprime = (moduleSeparation / alpaka::math::sin(acc, angleA + angleB)) * alpaka::math::sin(acc, angleA);

    // Compute arctan of the slope and take care of the slope = infinity case
    absArctanSlope = ((slope != kVerticalModuleSlope) ? fabs(alpaka::math::atan(acc, slope)) : kPi / 2.f);

    // Depending on which quadrant the pixel hit lies, we define the angleM by shifting them slightly differently
    if (xp > 0 and yp > 0) {
      angleM = absArctanSlope;
    } else if (xp > 0 and yp < 0) {
      angleM = kPi - absArctanSlope;
    } else if (xp < 0 and yp < 0) {
      angleM = kPi + absArctanSlope;
    } else  // if (xp < 0 and yp > 0)
    {
      angleM = 2.f * kPi - absArctanSlope;
    }

    // Then since the angleM sign is taken care of properly
    drprime_x = drprime * alpaka::math::sin(acc, angleM);
    drprime_y = drprime * alpaka::math::cos(acc, angleM);

    // The new anchor position is
    xa = xp + drprime_x;
    ya = yp + drprime_y;

    // Compute the new strip hit position (if the slope value is in special condition take care of the exceptions)
    if (slope ==
        kVerticalModuleSlope)  // Designated for tilted module when the slope is infinity (module lying along y-axis)
    {
      xn = xa;  // New x point is simply where the anchor is
      yn = yo;  // No shift in y
    } else if (slope == 0) {
      xn = xo;  // New x point is simply where the anchor is
      yn = ya;  // No shift in y
    } else {
      xn = (slope * xa + (1.f / slope) * xo - ya + yo) / (slope + (1.f / slope));  // new xn
      yn = (xn - xa) * slope + ya;                                                 // new yn
    }

    // Computing new Z position
    absdzprime = alpaka::math::abs(
        acc,
        moduleSeparation / alpaka::math::sin(acc, angleA + angleB) *
            alpaka::math::cos(
                acc,
                angleA));  // module separation sign is for shifting in radial direction for z-axis direction take care of the sign later

    // Depending on which one as closer to the interactin point compute the new z wrt to the pixel properly
    if (modules.moduleLayerType()[lowerModuleIndex] == Pixel) {
      abszn = alpaka::math::abs(acc, zp) + absdzprime;
    } else {
      abszn = alpaka::math::abs(acc, zp) - absdzprime;
    }

    zn = abszn * ((zp > 0) ? 1 : -1);  // Apply the sign of the zn

    shiftedCoords[0] = xn;
    shiftedCoords[1] = yn;
    shiftedCoords[2] = zn;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC bool runMiniDoubletDefaultAlgoBarrel(TAcc const& acc,
                                                     ModulesConst modules,
                                                     uint16_t lowerModuleIndex,
                                                     uint16_t upperModuleIndex,
                                                     unsigned int lowerHitIndex,
                                                     unsigned int upperHitIndex,
                                                     float& dz,
                                                     float& dPhi,
                                                     float& dPhiChange,
                                                     float& shiftedX,
                                                     float& shiftedY,
                                                     float& shiftedZ,
                                                     float& noShiftedDphi,
                                                     float& noShiftedDphiChange,
                                                     float xLower,
                                                     float yLower,
                                                     float zLower,
                                                     float rtLower,
                                                     float xUpper,
                                                     float yUpper,
                                                     float zUpper,
                                                     float rtUpper) {
    dz = zLower - zUpper;
    const float dzCut = modules.moduleType()[lowerModuleIndex] == PS ? 2.f : 10.f;
    const float sign = ((dz > 0) - (dz < 0)) * ((zLower > 0) - (zLower < 0));
    const float invertedcrossercut = (alpaka::math::abs(acc, dz) > 2) * sign;

    if ((alpaka::math::abs(acc, dz) >= dzCut) || (invertedcrossercut > 0))
      return false;

    float miniCut = 0;

    miniCut = modules.moduleLayerType()[lowerModuleIndex] == Pixel
                  ? dPhiThreshold(acc, rtLower, modules, lowerModuleIndex)
                  : dPhiThreshold(acc, rtUpper, modules, lowerModuleIndex);

    // Cut #2: dphi difference
    // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3085
    float xn = 0.f, yn = 0.f;  // , zn = 0;
    float shiftedRt2;
    if (modules.sides()[lowerModuleIndex] != Center)  // If barrel and not center it is tilted
    {
      // Shift the hits and calculate new xn, yn position
      float shiftedCoords[3];
      shiftStripHits(acc,
                     modules,
                     lowerModuleIndex,
                     upperModuleIndex,
                     lowerHitIndex,
                     upperHitIndex,
                     shiftedCoords,
                     xLower,
                     yLower,
                     zLower,
                     rtLower,
                     xUpper,
                     yUpper,
                     zUpper,
                     rtUpper);
      xn = shiftedCoords[0];
      yn = shiftedCoords[1];

      // Lower or the upper hit needs to be modified depending on which one was actually shifted
      if (modules.moduleLayerType()[lowerModuleIndex] == Pixel) {
        shiftedX = xn;
        shiftedY = yn;
        shiftedZ = zUpper;
        shiftedRt2 = xn * xn + yn * yn;

        dPhi = deltaPhi(acc, xLower, yLower, shiftedX, shiftedY);  //function from Hit.cc
        noShiftedDphi = deltaPhi(acc, xLower, yLower, xUpper, yUpper);
      } else {
        shiftedX = xn;
        shiftedY = yn;
        shiftedZ = zLower;
        shiftedRt2 = xn * xn + yn * yn;
        dPhi = deltaPhi(acc, shiftedX, shiftedY, xUpper, yUpper);
        noShiftedDphi = deltaPhi(acc, xLower, yLower, xUpper, yUpper);
      }
    } else {
      shiftedX = 0;
      shiftedY = 0;
      shiftedZ = 0;
      dPhi = deltaPhi(acc, xLower, yLower, xUpper, yUpper);
      noShiftedDphi = dPhi;
    }

    if (alpaka::math::abs(acc, dPhi) >= miniCut)
      return false;

    // Cut #3: The dphi change going from lower Hit to upper Hit
    // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3076
    if (modules.sides()[lowerModuleIndex] != Center) {
      // When it is tilted, use the new shifted positions
      // TODO: This is somewhat of an mystery.... somewhat confused why this is the case
      if (modules.moduleLayerType()[lowerModuleIndex] != Pixel) {
        // dPhi Change should be calculated so that the upper hit has higher rt.
        // In principle, this kind of check rt_lower < rt_upper should not be necessary because the hit shifting should have taken care of this.
        // (i.e. the strip hit is shifted to be aligned in the line of sight from interaction point to pixel hit of PS module guaranteeing rt ordering)
        // But I still placed this check for safety. (TODO: After checking explicitly if not needed remove later?)
        // setdeltaPhiChange(lowerHit.rt() < upperHitMod.rt() ? lowerHit.deltaPhiChange(upperHitMod) : upperHitMod.deltaPhiChange(lowerHit));

        dPhiChange = (rtLower * rtLower < shiftedRt2) ? deltaPhiChange(acc, xLower, yLower, shiftedX, shiftedY)
                                                      : deltaPhiChange(acc, shiftedX, shiftedY, xLower, yLower);
        noShiftedDphiChange = rtLower < rtUpper ? deltaPhiChange(acc, xLower, yLower, xUpper, yUpper)
                                                : deltaPhiChange(acc, xUpper, yUpper, xLower, yLower);
      } else {
        // dPhi Change should be calculated so that the upper hit has higher rt.
        // In principle, this kind of check rt_lower < rt_upper should not be necessary because the hit shifting should have taken care of this.
        // (i.e. the strip hit is shifted to be aligned in the line of sight from interaction point to pixel hit of PS module guaranteeing rt ordering)
        // But I still placed this check for safety. (TODO: After checking explicitly if not needed remove later?)

        dPhiChange = (shiftedRt2 < rtUpper * rtUpper) ? deltaPhiChange(acc, shiftedX, shiftedY, xUpper, yUpper)
                                                      : deltaPhiChange(acc, xUpper, yUpper, shiftedX, shiftedY);
        noShiftedDphiChange = rtLower < rtUpper ? deltaPhiChange(acc, xLower, yLower, xUpper, yUpper)
                                                : deltaPhiChange(acc, xUpper, yUpper, xLower, yLower);
      }
    } else {
      // When it is flat lying module, whichever is the lowerSide will always have rt lower
      dPhiChange = deltaPhiChange(acc, xLower, yLower, xUpper, yUpper);
      noShiftedDphiChange = dPhiChange;
    }

    return alpaka::math::abs(acc, dPhiChange) < miniCut;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC bool runMiniDoubletDefaultAlgoEndcap(TAcc const& acc,
                                                     ModulesConst modules,
                                                     uint16_t lowerModuleIndex,
                                                     uint16_t upperModuleIndex,
                                                     unsigned int lowerHitIndex,
                                                     unsigned int upperHitIndex,
                                                     float& drt,
                                                     float& dPhi,
                                                     float& dPhiChange,
                                                     float& shiftedX,
                                                     float& shiftedY,
                                                     float& shiftedZ,
                                                     float& noShiftedDphi,
                                                     float& noShiftedDphichange,
                                                     float xLower,
                                                     float yLower,
                                                     float zLower,
                                                     float rtLower,
                                                     float xUpper,
                                                     float yUpper,
                                                     float zUpper,
                                                     float rtUpper) {
    // There are series of cuts that applies to mini-doublet in a "endcap" region
    // Cut #1 : dz cut. The dz difference can't be larger than 1cm. (max separation is 4mm for modules in the endcap)
    // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3093
    // For PS module in case when it is tilted a different dz (after the strip hit shift) is calculated later.

    float dz = zLower - zUpper;  // Not const since later it might change depending on the type of module

    const float dzCut = 1.f;

    if (alpaka::math::abs(acc, dz) >= dzCut)
      return false;
    // Cut #2 : drt cut. The dz difference can't be larger than 1cm. (max separation is 4mm for modules in the endcap)
    // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3100
    const float drtCut = modules.moduleType()[lowerModuleIndex] == PS ? 2.f : 10.f;
    drt = rtLower - rtUpper;
    if (alpaka::math::abs(acc, drt) >= drtCut)
      return false;
    // The new scheme shifts strip hits to be "aligned" along the line of sight from interaction point to the pixel hit (if it is PS modules)
    float xn = 0, yn = 0, zn = 0;

    float shiftedCoords[3];
    shiftStripHits(acc,
                   modules,
                   lowerModuleIndex,
                   upperModuleIndex,
                   lowerHitIndex,
                   upperHitIndex,
                   shiftedCoords,
                   xLower,
                   yLower,
                   zLower,
                   rtLower,
                   xUpper,
                   yUpper,
                   zUpper,
                   rtUpper);

    xn = shiftedCoords[0];
    yn = shiftedCoords[1];
    zn = shiftedCoords[2];

    if (modules.moduleType()[lowerModuleIndex] == PS) {
      // Appropriate lower or upper hit is modified after checking which one was actually shifted
      if (modules.moduleLayerType()[lowerModuleIndex] == Pixel) {
        shiftedX = xn;
        shiftedY = yn;
        shiftedZ = zUpper;
        dPhi = deltaPhi(acc, xLower, yLower, shiftedX, shiftedY);
        noShiftedDphi = deltaPhi(acc, xLower, yLower, xUpper, yUpper);
      } else {
        shiftedX = xn;
        shiftedY = yn;
        shiftedZ = zLower;
        dPhi = deltaPhi(acc, shiftedX, shiftedY, xUpper, yUpper);
        noShiftedDphi = deltaPhi(acc, xLower, yLower, xUpper, yUpper);
      }
    } else {
      shiftedX = xn;
      shiftedY = yn;
      shiftedZ = zUpper;
      dPhi = deltaPhi(acc, xLower, yLower, xn, yn);
      noShiftedDphi = deltaPhi(acc, xLower, yLower, xUpper, yUpper);
    }

    // dz needs to change if it is a PS module where the strip hits are shifted in order to properly account for the case when a tilted module falls under "endcap logic"
    // if it was an endcap it will have zero effect
    if (modules.moduleType()[lowerModuleIndex] == PS) {
      dz = modules.moduleLayerType()[lowerModuleIndex] == Pixel ? zLower - zn : zUpper - zn;
    }

    float miniCut = 0;
    miniCut = modules.moduleLayerType()[lowerModuleIndex] == Pixel
                  ? dPhiThreshold(acc, rtLower, modules, lowerModuleIndex, dPhi, dz)
                  : dPhiThreshold(acc, rtUpper, modules, lowerModuleIndex, dPhi, dz);

    if (alpaka::math::abs(acc, dPhi) >= miniCut)
      return false;

    // Cut #4: Another cut on the dphi after some modification
    // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3119-L3124

    float dzFrac = alpaka::math::abs(acc, dz) / alpaka::math::abs(acc, zLower);
    dPhiChange = dPhi / dzFrac * (1.f + dzFrac);
    noShiftedDphichange = noShiftedDphi / dzFrac * (1.f + dzFrac);

    return alpaka::math::abs(acc, dPhiChange) < miniCut;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC bool runMiniDoubletDefaultAlgo(TAcc const& acc,
                                               ModulesConst modules,
                                               uint16_t lowerModuleIndex,
                                               uint16_t upperModuleIndex,
                                               unsigned int lowerHitIndex,
                                               unsigned int upperHitIndex,
                                               float& dz,
                                               float& dPhi,
                                               float& dPhiChange,
                                               float& shiftedX,
                                               float& shiftedY,
                                               float& shiftedZ,
                                               float& noShiftedDphi,
                                               float& noShiftedDphiChange,
                                               float xLower,
                                               float yLower,
                                               float zLower,
                                               float rtLower,
                                               float xUpper,
                                               float yUpper,
                                               float zUpper,
                                               float rtUpper) {
    if (modules.subdets()[lowerModuleIndex] == Barrel) {
      return runMiniDoubletDefaultAlgoBarrel(acc,
                                             modules,
                                             lowerModuleIndex,
                                             upperModuleIndex,
                                             lowerHitIndex,
                                             upperHitIndex,
                                             dz,
                                             dPhi,
                                             dPhiChange,
                                             shiftedX,
                                             shiftedY,
                                             shiftedZ,
                                             noShiftedDphi,
                                             noShiftedDphiChange,
                                             xLower,
                                             yLower,
                                             zLower,
                                             rtLower,
                                             xUpper,
                                             yUpper,
                                             zUpper,
                                             rtUpper);
    } else {
      return runMiniDoubletDefaultAlgoEndcap(acc,
                                             modules,
                                             lowerModuleIndex,
                                             upperModuleIndex,
                                             lowerHitIndex,
                                             upperHitIndex,
                                             dz,
                                             dPhi,
                                             dPhiChange,
                                             shiftedX,
                                             shiftedY,
                                             shiftedZ,
                                             noShiftedDphi,
                                             noShiftedDphiChange,
                                             xLower,
                                             yLower,
                                             zLower,
                                             rtLower,
                                             xUpper,
                                             yUpper,
                                             zUpper,
                                             rtUpper);
    }
  }

  struct CreateMiniDoublets {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  ModulesConst modules,
                                  HitsConst hits,
                                  HitsRangesConst hitsRanges,
                                  MiniDoublets mds,
                                  MiniDoubletsOccupancy mdsOccupancy,
                                  ObjectRangesConst ranges) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (uint16_t lowerModuleIndex = globalThreadIdx[1]; lowerModuleIndex < modules.nLowerModules();
           lowerModuleIndex += gridThreadExtent[1]) {
        uint16_t upperModuleIndex = modules.partnerModuleIndices()[lowerModuleIndex];
        int nLowerHits = hitsRanges.hitRangesnLower()[lowerModuleIndex];
        int nUpperHits = hitsRanges.hitRangesnUpper()[lowerModuleIndex];
        if (hitsRanges.hitRangesLower()[lowerModuleIndex] == -1)
          continue;
        unsigned int upHitArrayIndex = hitsRanges.hitRangesUpper()[lowerModuleIndex];
        unsigned int loHitArrayIndex = hitsRanges.hitRangesLower()[lowerModuleIndex];
        int limit = nUpperHits * nLowerHits;

        for (int hitIndex = globalThreadIdx[2]; hitIndex < limit; hitIndex += gridThreadExtent[2]) {
          int lowerHitIndex = hitIndex / nUpperHits;
          int upperHitIndex = hitIndex % nUpperHits;
          if (upperHitIndex >= nUpperHits)
            continue;
          if (lowerHitIndex >= nLowerHits)
            continue;
          unsigned int lowerHitArrayIndex = loHitArrayIndex + lowerHitIndex;
          float xLower = hits.xs()[lowerHitArrayIndex];
          float yLower = hits.ys()[lowerHitArrayIndex];
          float zLower = hits.zs()[lowerHitArrayIndex];
          float rtLower = hits.rts()[lowerHitArrayIndex];
          unsigned int upperHitArrayIndex = upHitArrayIndex + upperHitIndex;
          float xUpper = hits.xs()[upperHitArrayIndex];
          float yUpper = hits.ys()[upperHitArrayIndex];
          float zUpper = hits.zs()[upperHitArrayIndex];
          float rtUpper = hits.rts()[upperHitArrayIndex];

          float dz, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDphi, noShiftedDphiChange;
          bool success = runMiniDoubletDefaultAlgo(acc,
                                                   modules,
                                                   lowerModuleIndex,
                                                   upperModuleIndex,
                                                   lowerHitArrayIndex,
                                                   upperHitArrayIndex,
                                                   dz,
                                                   dphi,
                                                   dphichange,
                                                   shiftedX,
                                                   shiftedY,
                                                   shiftedZ,
                                                   noShiftedDphi,
                                                   noShiftedDphiChange,
                                                   xLower,
                                                   yLower,
                                                   zLower,
                                                   rtLower,
                                                   xUpper,
                                                   yUpper,
                                                   zUpper,
                                                   rtUpper);
          if (success) {
            int totOccupancyMDs = alpaka::atomicAdd(
                acc, &mdsOccupancy.totOccupancyMDs()[lowerModuleIndex], 1u, alpaka::hierarchy::Threads{});
            if (totOccupancyMDs >= (ranges.miniDoubletModuleOccupancy()[lowerModuleIndex])) {
#ifdef WARNINGS
              printf("Mini-doublet excess alert! Module index =  %d\n", lowerModuleIndex);
#endif
            } else {
              int mdModuleIndex =
                  alpaka::atomicAdd(acc, &mdsOccupancy.nMDs()[lowerModuleIndex], 1u, alpaka::hierarchy::Threads{});
              unsigned int mdIndex = ranges.miniDoubletModuleIndices()[lowerModuleIndex] + mdModuleIndex;

              addMDToMemory(acc,
                            mds,
                            hits,
                            modules,
                            lowerHitArrayIndex,
                            upperHitArrayIndex,
                            lowerModuleIndex,
                            dz,
                            dphi,
                            dphichange,
                            shiftedX,
                            shiftedY,
                            shiftedZ,
                            noShiftedDphi,
                            noShiftedDphiChange,
                            mdIndex);
            }
          }
        }
      }
    }
  };

  struct CreateMDArrayRangesGPU {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, ModulesConst modules, ObjectRanges ranges) const {
      // implementation is 1D with a single block
      static_assert(std::is_same_v<TAcc, ALPAKA_ACCELERATOR_NAMESPACE::Acc1D>, "Should be Acc1D");
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));

      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      // Declare variables in shared memory and set to 0
      int& nTotalMDs = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      if (cms::alpakatools::once_per_block(acc)) {
        nTotalMDs = 0;
      }
      alpaka::syncBlockThreads(acc);

      for (uint16_t i = globalThreadIdx[0]; i < modules.nLowerModules(); i += gridThreadExtent[0]) {
        short module_rings = modules.rings()[i];
        short module_layers = modules.layers()[i];
        short module_subdets = modules.subdets()[i];
        float module_eta = alpaka::math::abs(acc, modules.eta()[i]);

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
          occupancy = 49;
        else if (category_number == 0 && eta_number == 1)
          occupancy = 42;
        else if (category_number == 0 && eta_number == 2)
          occupancy = 37;
        else if (category_number == 0 && eta_number == 3)
          occupancy = 41;
        else if (category_number == 1)
          occupancy = 100;
        else if (category_number == 2 && eta_number == 1)
          occupancy = 16;
        else if (category_number == 2 && eta_number == 2)
          occupancy = 19;
        else if (category_number == 3 && eta_number == 1)
          occupancy = 14;
        else if (category_number == 3 && eta_number == 2)
          occupancy = 20;
        else if (category_number == 3 && eta_number == 3)
          occupancy = 25;
        else {
          occupancy = 0;
#ifdef WARNINGS
          printf("Unhandled case in createMDArrayRangesGPU! Module index = %i\n", i);
#endif
        }

        unsigned int nTotMDs = alpaka::atomicAdd(acc, &nTotalMDs, occupancy, alpaka::hierarchy::Threads{});

        ranges.miniDoubletModuleIndices()[i] = nTotMDs;
        ranges.miniDoubletModuleOccupancy()[i] = occupancy;
      }

      // Wait for all threads to finish before reporting final values
      alpaka::syncBlockThreads(acc);
      if (cms::alpakatools::once_per_block(acc)) {
        ranges.miniDoubletModuleIndices()[modules.nLowerModules()] = nTotalMDs;
        ranges.nTotalMDs() = nTotalMDs;
      }
    }
  };

  struct AddMiniDoubletRangesToEventExplicit {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  ModulesConst modules,
                                  MiniDoubletsOccupancy mdsOccupancy,
                                  ObjectRanges ranges,
                                  HitsRangesConst hitsRanges) const {
      // implementation is 1D with a single block
      static_assert(std::is_same_v<TAcc, ALPAKA_ACCELERATOR_NAMESPACE::Acc1D>, "Should be Acc1D");
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));

      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (uint16_t i = globalThreadIdx[0]; i < modules.nLowerModules(); i += gridThreadExtent[0]) {
        if (mdsOccupancy.nMDs()[i] == 0 or hitsRanges.hitRanges()[i][0] == -1) {
          ranges.mdRanges()[i][0] = -1;
          ranges.mdRanges()[i][1] = -1;
        } else {
          ranges.mdRanges()[i][0] = ranges.miniDoubletModuleIndices()[i];
          ranges.mdRanges()[i][1] = ranges.miniDoubletModuleIndices()[i] + mdsOccupancy.nMDs()[i] - 1;
        }
      }
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif
