#ifndef RecoTracker_LSTCore_src_alpaka_Kernels_h
#define RecoTracker_LSTCore_src_alpaka_Kernels_h

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/ModulesSoA.h"
#include "RecoTracker/LSTCore/interface/ObjectRangesSoA.h"
#include "RecoTracker/LSTCore/interface/MiniDoubletsSoA.h"
#include "RecoTracker/LSTCore/interface/PixelQuintupletsSoA.h"
#include "RecoTracker/LSTCore/interface/PixelTripletsSoA.h"
#include "RecoTracker/LSTCore/interface/QuintupletsSoA.h"
#include "RecoTracker/LSTCore/interface/SegmentsSoA.h"
#include "RecoTracker/LSTCore/interface/TripletsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void rmQuintupletFromMemory(Quintuplets quintuplets,
                                                             unsigned int quintupletIndex,
                                                             bool secondpass = false) {
    quintuplets.isDup()[quintupletIndex] |= 1 + secondpass;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void rmPixelTripletFromMemory(PixelTriplets pixelTriplets,
                                                               unsigned int pixelTripletIndex) {
    pixelTriplets.isDup()[pixelTripletIndex] = true;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void rmPixelQuintupletFromMemory(PixelQuintuplets pixelQuintuplets,
                                                                  unsigned int pixelQuintupletIndex) {
    pixelQuintuplets.isDup()[pixelQuintupletIndex] = true;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void rmPixelSegmentFromMemory(SegmentsPixel segmentsPixel,
                                                               unsigned int pixelSegmentArrayIndex,
                                                               bool secondpass = false) {
    segmentsPixel.isDup()[pixelSegmentArrayIndex] |= 1 + secondpass;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE int checkHitsT5(unsigned int ix, unsigned int jx, QuintupletsConst quintuplets) {
    unsigned int hits1[Params_T5::kHits];
    unsigned int hits2[Params_T5::kHits];

    for (int i = 0; i < Params_T5::kHits; i++) {
      hits1[i] = quintuplets.hitIndices()[ix][i];
      hits2[i] = quintuplets.hitIndices()[jx][i];
    }

    int nMatched = 0;
    for (int i = 0; i < Params_T5::kHits; i++) {
      bool matched = false;
      for (int j = 0; j < Params_T5::kHits; j++) {
        if (hits1[i] == hits2[j]) {
          matched = true;
          break;
        }
      }
      if (matched) {
        nMatched++;
      }
    }
    return nMatched;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE int checkHitspT5(unsigned int ix,
                                                  unsigned int jx,
                                                  PixelQuintupletsConst pixelQuintuplets) {
    unsigned int hits1[Params_pT5::kHits];
    unsigned int hits2[Params_pT5::kHits];

    for (int i = 0; i < Params_pT5::kHits; i++) {
      hits1[i] = pixelQuintuplets.hitIndices()[ix][i];
      hits2[i] = pixelQuintuplets.hitIndices()[jx][i];
    }

    int nMatched = 0;
    for (int i = 0; i < Params_pT5::kHits; i++) {
      bool matched = false;
      for (int j = 0; j < Params_pT5::kHits; j++) {
        if (hits1[i] == hits2[j]) {
          matched = true;
          break;
        }
      }
      if (matched) {
        nMatched++;
      }
    }
    return nMatched;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void checkHitspT3(unsigned int ix,
                                                   unsigned int jx,
                                                   PixelTripletsConst pixelTriplets,
                                                   int* matched) {
    int phits1[Params_pLS::kHits];
    int phits2[Params_pLS::kHits];

    for (int i = 0; i < Params_pLS::kHits; i++) {
      phits1[i] = pixelTriplets.hitIndices()[ix][i];
      phits2[i] = pixelTriplets.hitIndices()[jx][i];
    }

    int npMatched = 0;
    for (int i = 0; i < Params_pLS::kHits; i++) {
      bool pmatched = false;
      for (int j = 0; j < Params_pLS::kHits; j++) {
        if (phits1[i] == phits2[j]) {
          pmatched = true;
          break;
        }
      }
      if (pmatched) {
        npMatched++;
      }
    }

    int hits1[Params_T3::kHits];
    int hits2[Params_T3::kHits];

    for (int i = 0; i < Params_T3::kHits; i++) {
      hits1[i] = pixelTriplets.hitIndices()[ix][i + 4];  // Omitting the pLS hits
      hits2[i] = pixelTriplets.hitIndices()[jx][i + 4];  // Omitting the pLS hits
    }

    int nMatched = 0;
    for (int i = 0; i < Params_T3::kHits; i++) {
      bool tmatched = false;
      for (int j = 0; j < Params_T3::kHits; j++) {
        if (hits1[i] == hits2[j]) {
          tmatched = true;
          break;
        }
      }
      if (tmatched) {
        nMatched++;
      }
    }

    matched[0] = npMatched;
    matched[1] = nMatched;
  }

  struct RemoveDupQuintupletsAfterBuild {
    ALPAKA_FN_ACC void operator()(Acc3D const& acc,
                                  ModulesConst modules,
                                  Quintuplets quintuplets,
                                  QuintupletsOccupancyConst quintupletsOccupancy,
                                  ObjectRangesConst ranges) const {
      for (unsigned int lowmod : cms::alpakatools::uniform_elements_z(acc, modules.nLowerModules())) {
        unsigned int nQuintuplets_lowmod = quintupletsOccupancy.nQuintuplets()[lowmod];
        int quintupletModuleIndices_lowmod = ranges.quintupletModuleIndices()[lowmod];

        for (unsigned int ix1 : cms::alpakatools::uniform_elements_y(acc, nQuintuplets_lowmod)) {
          unsigned int ix = quintupletModuleIndices_lowmod + ix1;
          float eta1 = __H2F(quintuplets.eta()[ix]);
          float phi1 = __H2F(quintuplets.phi()[ix]);
          float score_rphisum1 = __H2F(quintuplets.score_rphisum()[ix]);

          for (unsigned int jx1 : cms::alpakatools::uniform_elements_x(acc, ix1 + 1, nQuintuplets_lowmod)) {
            unsigned int jx = quintupletModuleIndices_lowmod + jx1;

            float eta2 = __H2F(quintuplets.eta()[jx]);
            float phi2 = __H2F(quintuplets.phi()[jx]);
            float dEta = alpaka::math::abs(acc, eta1 - eta2);
            float dPhi = calculate_dPhi(phi1, phi2);
            float score_rphisum2 = __H2F(quintuplets.score_rphisum()[jx]);

            if (dEta > 0.1f)
              continue;

            if (alpaka::math::abs(acc, dPhi) > 0.1f)
              continue;

            int nMatched = checkHitsT5(ix, jx, quintuplets);
            const int minNHitsForDup_T5 = 7;
            if (nMatched >= minNHitsForDup_T5) {
              if (score_rphisum1 >= score_rphisum2) {
                rmQuintupletFromMemory(quintuplets, ix);
              } else {
                rmQuintupletFromMemory(quintuplets, jx);
              }
            }
          }
        }
      }
    }
  };

  struct RemoveDupQuintupletsBeforeTC {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  Quintuplets quintuplets,
                                  QuintupletsOccupancyConst quintupletsOccupancy,
                                  ObjectRangesConst ranges) const {
      for (unsigned int lowmodIdx1 : cms::alpakatools::uniform_elements_y(acc, ranges.nEligibleT5Modules())) {
        uint16_t lowmod1 = ranges.indicesOfEligibleT5Modules()[lowmodIdx1];
        unsigned int nQuintuplets_lowmod1 = quintupletsOccupancy.nQuintuplets()[lowmod1];
        if (nQuintuplets_lowmod1 == 0)
          continue;

        unsigned int quintupletModuleIndices_lowmod1 = ranges.quintupletModuleIndices()[lowmod1];

        for (unsigned int lowmodIdx2 :
             cms::alpakatools::uniform_elements_x(acc, lowmodIdx1, ranges.nEligibleT5Modules())) {
          uint16_t lowmod2 = ranges.indicesOfEligibleT5Modules()[lowmodIdx2];
          unsigned int nQuintuplets_lowmod2 = quintupletsOccupancy.nQuintuplets()[lowmod2];
          if (nQuintuplets_lowmod2 == 0)
            continue;

          unsigned int quintupletModuleIndices_lowmod2 = ranges.quintupletModuleIndices()[lowmod2];

          for (unsigned int ix1 = 0; ix1 < nQuintuplets_lowmod1; ix1 += 1) {
            unsigned int ix = quintupletModuleIndices_lowmod1 + ix1;
            if (quintuplets.isDup()[ix] & 1)
              continue;

            bool isPT5_ix = quintuplets.partOfPT5()[ix];

            for (unsigned int jx1 = 0; jx1 < nQuintuplets_lowmod2; jx1++) {
              unsigned int jx = quintupletModuleIndices_lowmod2 + jx1;
              if (ix == jx)
                continue;

              if (quintuplets.isDup()[jx] & 1)
                continue;

              bool isPT5_jx = quintuplets.partOfPT5()[jx];

              if (isPT5_ix && isPT5_jx)
                continue;

              float eta1 = __H2F(quintuplets.eta()[ix]);
              float phi1 = __H2F(quintuplets.phi()[ix]);
              float score_rphisum1 = __H2F(quintuplets.score_rphisum()[ix]);

              float eta2 = __H2F(quintuplets.eta()[jx]);
              float phi2 = __H2F(quintuplets.phi()[jx]);
              float score_rphisum2 = __H2F(quintuplets.score_rphisum()[jx]);

              float dEta = alpaka::math::abs(acc, eta1 - eta2);
              float dPhi = calculate_dPhi(phi1, phi2);

              if (dEta > 0.1f)
                continue;

              if (alpaka::math::abs(acc, dPhi) > 0.1f)
                continue;

              float dR2 = dEta * dEta + dPhi * dPhi;
              int nMatched = checkHitsT5(ix, jx, quintuplets);
              const int minNHitsForDup_T5 = 5;
              if (dR2 < 0.001f || nMatched >= minNHitsForDup_T5) {
                if (isPT5_jx || score_rphisum1 > score_rphisum2) {
                  rmQuintupletFromMemory(quintuplets, ix, true);
                } else if (isPT5_ix || score_rphisum1 < score_rphisum2) {
                  rmQuintupletFromMemory(quintuplets, jx, true);
                } else {
                  rmQuintupletFromMemory(quintuplets, (ix < jx ? ix : jx), true);
                }
              }
            }
          }
        }
      }
    }
  };

  struct RemoveDupPixelTripletsFromMap {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc, PixelTriplets pixelTriplets) const {
      for (unsigned int ix : cms::alpakatools::uniform_elements_y(acc, pixelTriplets.nPixelTriplets())) {
        for (unsigned int jx : cms::alpakatools::uniform_elements_x(acc, pixelTriplets.nPixelTriplets())) {
          if (ix == jx)
            continue;

          int nMatched[2];
          checkHitspT3(ix, jx, pixelTriplets, nMatched);
          const int minNHitsForDup_pT3 = 5;
          if ((nMatched[0] + nMatched[1]) >= minNHitsForDup_pT3) {
            // Check the layers
            if (pixelTriplets.logicalLayers()[jx][2] < pixelTriplets.logicalLayers()[ix][2]) {
              rmPixelTripletFromMemory(pixelTriplets, ix);
              break;
            } else if (pixelTriplets.logicalLayers()[ix][2] == pixelTriplets.logicalLayers()[jx][2] &&
                       __H2F(pixelTriplets.score()[ix]) > __H2F(pixelTriplets.score()[jx])) {
              rmPixelTripletFromMemory(pixelTriplets, ix);
              break;
            } else if (pixelTriplets.logicalLayers()[ix][2] == pixelTriplets.logicalLayers()[jx][2] &&
                       (__H2F(pixelTriplets.score()[ix]) == __H2F(pixelTriplets.score()[jx])) && (ix < jx)) {
              rmPixelTripletFromMemory(pixelTriplets, ix);
              break;
            }
          }
        }
      }
    }
  };

  struct RemoveDupPixelQuintupletsFromMap {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc, PixelQuintuplets pixelQuintuplets) const {
      unsigned int nPixelQuintuplets = pixelQuintuplets.nPixelQuintuplets();
      for (unsigned int ix : cms::alpakatools::uniform_elements_y(acc, nPixelQuintuplets)) {
        float score1 = __H2F(pixelQuintuplets.score()[ix]);
        for (unsigned int jx : cms::alpakatools::uniform_elements_x(acc, nPixelQuintuplets)) {
          if (ix == jx)
            continue;

          int nMatched = checkHitspT5(ix, jx, pixelQuintuplets);
          float score2 = __H2F(pixelQuintuplets.score()[jx]);
          const int minNHitsForDup_pT5 = 7;
          if (nMatched >= minNHitsForDup_pT5) {
            if (score1 > score2 or ((score1 == score2) and (ix > jx))) {
              rmPixelQuintupletFromMemory(pixelQuintuplets, ix);
              break;
            }
          }
        }
      }
    }
  };

  struct CheckHitspLS {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  ModulesConst modules,
                                  SegmentsOccupancyConst segmentsOccupancy,
                                  SegmentsPixel segmentsPixel,
                                  bool secondpass) const {
      int pixelModuleIndex = modules.nLowerModules();
      unsigned int nPixelSegments = segmentsOccupancy.nSegments()[pixelModuleIndex];

      if (nPixelSegments > n_max_pixel_segments_per_module)
        nPixelSegments = n_max_pixel_segments_per_module;

      for (unsigned int ix : cms::alpakatools::uniform_elements_y(acc, nPixelSegments)) {
        if (secondpass && (!segmentsPixel.isQuad()[ix] || (segmentsPixel.isDup()[ix] & 1)))
          continue;

        unsigned int phits1[Params_pLS::kHits];
        phits1[0] = segmentsPixel.pLSHitsIdxs()[ix].x;
        phits1[1] = segmentsPixel.pLSHitsIdxs()[ix].y;
        phits1[2] = segmentsPixel.pLSHitsIdxs()[ix].z;
        phits1[3] = segmentsPixel.pLSHitsIdxs()[ix].w;
        float eta_pix1 = segmentsPixel.eta()[ix];
        float phi_pix1 = segmentsPixel.phi()[ix];

        for (unsigned int jx : cms::alpakatools::uniform_elements_x(acc, ix + 1, nPixelSegments)) {
          float eta_pix2 = segmentsPixel.eta()[jx];
          float phi_pix2 = segmentsPixel.phi()[jx];

          if (alpaka::math::abs(acc, eta_pix2 - eta_pix1) > 0.1f)
            continue;

          if (secondpass && (!segmentsPixel.isQuad()[jx] || (segmentsPixel.isDup()[jx] & 1)))
            continue;

          int8_t quad_diff = segmentsPixel.isQuad()[ix] - segmentsPixel.isQuad()[jx];
          float score_diff = segmentsPixel.score()[ix] - segmentsPixel.score()[jx];
          // Always keep quads over trips. If they are the same, we want the object with better score
          int idxToRemove;
          if (quad_diff > 0)
            idxToRemove = jx;
          else if (quad_diff < 0)
            idxToRemove = ix;
          else if (score_diff < 0)
            idxToRemove = jx;
          else if (score_diff > 0)
            idxToRemove = ix;
          else
            idxToRemove = ix;

          unsigned int phits2[Params_pLS::kHits];
          phits2[0] = segmentsPixel.pLSHitsIdxs()[jx].x;
          phits2[1] = segmentsPixel.pLSHitsIdxs()[jx].y;
          phits2[2] = segmentsPixel.pLSHitsIdxs()[jx].z;
          phits2[3] = segmentsPixel.pLSHitsIdxs()[jx].w;

          int npMatched = 0;
          for (int i = 0; i < Params_pLS::kHits; i++) {
            bool pmatched = false;
            for (int j = 0; j < Params_pLS::kHits; j++) {
              if (phits1[i] == phits2[j]) {
                pmatched = true;
                break;
              }
            }
            if (pmatched) {
              npMatched++;
              // Only one hit is enough
              if (secondpass)
                break;
            }
          }
          const int minNHitsForDup_pLS = 3;
          if (npMatched >= minNHitsForDup_pLS) {
            rmPixelSegmentFromMemory(segmentsPixel, idxToRemove, secondpass);
          }
          if (secondpass) {
            float dEta = alpaka::math::abs(acc, eta_pix1 - eta_pix2);
            float dPhi = calculate_dPhi(phi_pix1, phi_pix2);

            float dR2 = dEta * dEta + dPhi * dPhi;
            if ((npMatched >= 1) || (dR2 < 1e-5f)) {
              rmPixelSegmentFromMemory(segmentsPixel, idxToRemove, secondpass);
            }
          }
        }
      }
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
