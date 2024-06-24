#ifndef Kernels_cuh
#define Kernels_cuh

#ifdef LST_IS_CMSSW_PACKAGE
#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#include "RecoTracker/LSTCore/interface/alpaka/Module.h"
#else
#include "Constants.h"
#include "Module.h"
#endif

#include "Hit.h"
#include "MiniDoublet.h"
#include "Segment.h"
#include "Triplet.h"
#include "Quintuplet.h"
#include "PixelTriplet.h"

namespace SDL {
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void rmQuintupletFromMemory(struct SDL::quintuplets& quintupletsInGPU,
                                                             unsigned int quintupletIndex,
                                                             bool secondpass = false) {
    quintupletsInGPU.isDup[quintupletIndex] |= 1 + secondpass;
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void rmPixelTripletFromMemory(struct SDL::pixelTriplets& pixelTripletsInGPU,
                                                               unsigned int pixelTripletIndex) {
    pixelTripletsInGPU.isDup[pixelTripletIndex] = true;
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void rmPixelQuintupletFromMemory(struct SDL::pixelQuintuplets& pixelQuintupletsInGPU,
                                                                  unsigned int pixelQuintupletIndex) {
    pixelQuintupletsInGPU.isDup[pixelQuintupletIndex] = true;
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void rmPixelSegmentFromMemory(struct SDL::segments& segmentsInGPU,
                                                               unsigned int pixelSegmentArrayIndex,
                                                               bool secondpass = false) {
    segmentsInGPU.isDup[pixelSegmentArrayIndex] |= 1 + secondpass;
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE int checkHitsT5(unsigned int ix,
                                                 unsigned int jx,
                                                 struct SDL::quintuplets& quintupletsInGPU) {
    unsigned int hits1[10];
    unsigned int hits2[10];

    hits1[0] = quintupletsInGPU.hitIndices[10 * ix];
    hits1[1] = quintupletsInGPU.hitIndices[10 * ix + 1];
    hits1[2] = quintupletsInGPU.hitIndices[10 * ix + 2];
    hits1[3] = quintupletsInGPU.hitIndices[10 * ix + 3];
    hits1[4] = quintupletsInGPU.hitIndices[10 * ix + 4];
    hits1[5] = quintupletsInGPU.hitIndices[10 * ix + 5];
    hits1[6] = quintupletsInGPU.hitIndices[10 * ix + 6];
    hits1[7] = quintupletsInGPU.hitIndices[10 * ix + 7];
    hits1[8] = quintupletsInGPU.hitIndices[10 * ix + 8];
    hits1[9] = quintupletsInGPU.hitIndices[10 * ix + 9];

    hits2[0] = quintupletsInGPU.hitIndices[10 * jx];
    hits2[1] = quintupletsInGPU.hitIndices[10 * jx + 1];
    hits2[2] = quintupletsInGPU.hitIndices[10 * jx + 2];
    hits2[3] = quintupletsInGPU.hitIndices[10 * jx + 3];
    hits2[4] = quintupletsInGPU.hitIndices[10 * jx + 4];
    hits2[5] = quintupletsInGPU.hitIndices[10 * jx + 5];
    hits2[6] = quintupletsInGPU.hitIndices[10 * jx + 6];
    hits2[7] = quintupletsInGPU.hitIndices[10 * jx + 7];
    hits2[8] = quintupletsInGPU.hitIndices[10 * jx + 8];
    hits2[9] = quintupletsInGPU.hitIndices[10 * jx + 9];

    int nMatched = 0;
    for (int i = 0; i < 10; i++) {
      bool matched = false;
      for (int j = 0; j < 10; j++) {
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
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE int checkHitspT5(unsigned int ix,
                                                  unsigned int jx,
                                                  struct SDL::pixelQuintuplets& pixelQuintupletsInGPU) {
    unsigned int hits1[14];
    unsigned int hits2[14];

    hits1[0] = pixelQuintupletsInGPU.hitIndices[14 * ix];
    hits1[1] = pixelQuintupletsInGPU.hitIndices[14 * ix + 1];
    hits1[2] = pixelQuintupletsInGPU.hitIndices[14 * ix + 2];
    hits1[3] = pixelQuintupletsInGPU.hitIndices[14 * ix + 3];
    hits1[4] = pixelQuintupletsInGPU.hitIndices[14 * ix + 4];
    hits1[5] = pixelQuintupletsInGPU.hitIndices[14 * ix + 5];
    hits1[6] = pixelQuintupletsInGPU.hitIndices[14 * ix + 6];
    hits1[7] = pixelQuintupletsInGPU.hitIndices[14 * ix + 7];
    hits1[8] = pixelQuintupletsInGPU.hitIndices[14 * ix + 8];
    hits1[9] = pixelQuintupletsInGPU.hitIndices[14 * ix + 9];
    hits1[10] = pixelQuintupletsInGPU.hitIndices[14 * ix + 10];
    hits1[11] = pixelQuintupletsInGPU.hitIndices[14 * ix + 11];
    hits1[12] = pixelQuintupletsInGPU.hitIndices[14 * ix + 12];
    hits1[13] = pixelQuintupletsInGPU.hitIndices[14 * ix + 13];

    hits2[0] = pixelQuintupletsInGPU.hitIndices[14 * jx];
    hits2[1] = pixelQuintupletsInGPU.hitIndices[14 * jx + 1];
    hits2[2] = pixelQuintupletsInGPU.hitIndices[14 * jx + 2];
    hits2[3] = pixelQuintupletsInGPU.hitIndices[14 * jx + 3];
    hits2[4] = pixelQuintupletsInGPU.hitIndices[14 * jx + 4];
    hits2[5] = pixelQuintupletsInGPU.hitIndices[14 * jx + 5];
    hits2[6] = pixelQuintupletsInGPU.hitIndices[14 * jx + 6];
    hits2[7] = pixelQuintupletsInGPU.hitIndices[14 * jx + 7];
    hits2[8] = pixelQuintupletsInGPU.hitIndices[14 * jx + 8];
    hits2[9] = pixelQuintupletsInGPU.hitIndices[14 * jx + 9];
    hits2[10] = pixelQuintupletsInGPU.hitIndices[14 * jx + 10];
    hits2[11] = pixelQuintupletsInGPU.hitIndices[14 * jx + 11];
    hits2[12] = pixelQuintupletsInGPU.hitIndices[14 * jx + 12];
    hits2[13] = pixelQuintupletsInGPU.hitIndices[14 * jx + 13];

    int nMatched = 0;
    for (int i = 0; i < 14; i++) {
      bool matched = false;
      for (int j = 0; j < 14; j++) {
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
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void checkHitspT3(unsigned int ix,
                                                   unsigned int jx,
                                                   struct SDL::pixelTriplets& pixelTripletsInGPU,
                                                   int* matched) {
    int phits1[4] = {-1, -1, -1, -1};
    int phits2[4] = {-1, -1, -1, -1};
    phits1[0] = pixelTripletsInGPU.hitIndices[10 * ix];
    phits1[1] = pixelTripletsInGPU.hitIndices[10 * ix + 1];
    phits1[2] = pixelTripletsInGPU.hitIndices[10 * ix + 2];
    phits1[3] = pixelTripletsInGPU.hitIndices[10 * ix + 3];

    phits2[0] = pixelTripletsInGPU.hitIndices[10 * jx];
    phits2[1] = pixelTripletsInGPU.hitIndices[10 * jx + 1];
    phits2[2] = pixelTripletsInGPU.hitIndices[10 * jx + 2];
    phits2[3] = pixelTripletsInGPU.hitIndices[10 * jx + 3];

    int npMatched = 0;
    for (int i = 0; i < 4; i++) {
      bool pmatched = false;
      for (int j = 0; j < 4; j++) {
        if (phits1[i] == phits2[j]) {
          pmatched = true;
          break;
        }
      }
      if (pmatched) {
        npMatched++;
      }
    }

    int hits1[6] = {-1, -1, -1, -1, -1, -1};
    int hits2[6] = {-1, -1, -1, -1, -1, -1};
    hits1[0] = pixelTripletsInGPU.hitIndices[10 * ix + 4];
    hits1[1] = pixelTripletsInGPU.hitIndices[10 * ix + 5];
    hits1[2] = pixelTripletsInGPU.hitIndices[10 * ix + 6];
    hits1[3] = pixelTripletsInGPU.hitIndices[10 * ix + 7];
    hits1[4] = pixelTripletsInGPU.hitIndices[10 * ix + 8];
    hits1[5] = pixelTripletsInGPU.hitIndices[10 * ix + 9];

    hits2[0] = pixelTripletsInGPU.hitIndices[10 * jx + 4];
    hits2[1] = pixelTripletsInGPU.hitIndices[10 * jx + 5];
    hits2[2] = pixelTripletsInGPU.hitIndices[10 * jx + 6];
    hits2[3] = pixelTripletsInGPU.hitIndices[10 * jx + 7];
    hits2[4] = pixelTripletsInGPU.hitIndices[10 * jx + 8];
    hits2[5] = pixelTripletsInGPU.hitIndices[10 * jx + 9];

    int nMatched = 0;
    for (int i = 0; i < 6; i++) {
      bool tmatched = false;
      for (int j = 0; j < 6; j++) {
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
  };

  struct removeDupQuintupletsInGPUAfterBuild {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  struct SDL::modules modulesInGPU,
                                  struct SDL::quintuplets quintupletsInGPU,
                                  struct SDL::objectRanges rangesInGPU) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (unsigned int lowmod = globalThreadIdx[0]; lowmod < *modulesInGPU.nLowerModules;
           lowmod += gridThreadExtent[0]) {
        unsigned int nQuintuplets_lowmod = quintupletsInGPU.nQuintuplets[lowmod];
        int quintupletModuleIndices_lowmod = rangesInGPU.quintupletModuleIndices[lowmod];

        for (unsigned int ix1 = globalThreadIdx[1]; ix1 < nQuintuplets_lowmod; ix1 += gridThreadExtent[1]) {
          unsigned int ix = quintupletModuleIndices_lowmod + ix1;
          float eta1 = __H2F(quintupletsInGPU.eta[ix]);
          float phi1 = __H2F(quintupletsInGPU.phi[ix]);
          float score_rphisum1 = __H2F(quintupletsInGPU.score_rphisum[ix]);

          for (unsigned int jx1 = globalThreadIdx[2] + ix1 + 1; jx1 < nQuintuplets_lowmod; jx1 += gridThreadExtent[2]) {
            unsigned int jx = quintupletModuleIndices_lowmod + jx1;

            float eta2 = __H2F(quintupletsInGPU.eta[jx]);
            float phi2 = __H2F(quintupletsInGPU.phi[jx]);
            float dEta = alpaka::math::abs(acc, eta1 - eta2);
            float dPhi = SDL::calculate_dPhi(phi1, phi2);
            float score_rphisum2 = __H2F(quintupletsInGPU.score_rphisum[jx]);

            if (dEta > 0.1f)
              continue;

            if (alpaka::math::abs(acc, dPhi) > 0.1f)
              continue;

            int nMatched = checkHitsT5(ix, jx, quintupletsInGPU);
            if (nMatched >= 7) {
              if (score_rphisum1 >= score_rphisum2) {
                rmQuintupletFromMemory(quintupletsInGPU, ix);
              } else {
                rmQuintupletFromMemory(quintupletsInGPU, jx);
              }
            }
          }
        }
      }
    }
  };

  struct removeDupQuintupletsInGPUBeforeTC {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  struct SDL::quintuplets quintupletsInGPU,
                                  struct SDL::objectRanges rangesInGPU) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (unsigned int lowmodIdx1 = globalThreadIdx[1]; lowmodIdx1 < *(rangesInGPU.nEligibleT5Modules);
           lowmodIdx1 += gridThreadExtent[1]) {
        uint16_t lowmod1 = rangesInGPU.indicesOfEligibleT5Modules[lowmodIdx1];
        unsigned int nQuintuplets_lowmod1 = quintupletsInGPU.nQuintuplets[lowmod1];
        if (nQuintuplets_lowmod1 == 0)
          continue;

        unsigned int quintupletModuleIndices_lowmod1 = rangesInGPU.quintupletModuleIndices[lowmod1];

        for (unsigned int lowmodIdx2 = globalThreadIdx[2] + lowmodIdx1; lowmodIdx2 < *(rangesInGPU.nEligibleT5Modules);
             lowmodIdx2 += gridThreadExtent[2]) {
          uint16_t lowmod2 = rangesInGPU.indicesOfEligibleT5Modules[lowmodIdx2];
          unsigned int nQuintuplets_lowmod2 = quintupletsInGPU.nQuintuplets[lowmod2];
          if (nQuintuplets_lowmod2 == 0)
            continue;

          unsigned int quintupletModuleIndices_lowmod2 = rangesInGPU.quintupletModuleIndices[lowmod2];

          for (unsigned int ix1 = 0; ix1 < nQuintuplets_lowmod1; ix1 += 1) {
            unsigned int ix = quintupletModuleIndices_lowmod1 + ix1;
            if (quintupletsInGPU.partOfPT5[ix] || (quintupletsInGPU.isDup[ix] & 1))
              continue;

            for (unsigned int jx1 = 0; jx1 < nQuintuplets_lowmod2; jx1++) {
              unsigned int jx = quintupletModuleIndices_lowmod2 + jx1;
              if (ix == jx)
                continue;

              if (quintupletsInGPU.partOfPT5[jx] || (quintupletsInGPU.isDup[jx] & 1))
                continue;

              float eta1 = __H2F(quintupletsInGPU.eta[ix]);
              float phi1 = __H2F(quintupletsInGPU.phi[ix]);
              float score_rphisum1 = __H2F(quintupletsInGPU.score_rphisum[ix]);

              float eta2 = __H2F(quintupletsInGPU.eta[jx]);
              float phi2 = __H2F(quintupletsInGPU.phi[jx]);
              float score_rphisum2 = __H2F(quintupletsInGPU.score_rphisum[jx]);

              float dEta = alpaka::math::abs(acc, eta1 - eta2);
              float dPhi = SDL::calculate_dPhi(phi1, phi2);

              if (dEta > 0.1f)
                continue;

              if (alpaka::math::abs(acc, dPhi) > 0.1f)
                continue;

              float dR2 = dEta * dEta + dPhi * dPhi;
              int nMatched = checkHitsT5(ix, jx, quintupletsInGPU);
              if (dR2 < 0.001f || nMatched >= 5) {
                if (score_rphisum1 > score_rphisum2) {
                  rmQuintupletFromMemory(quintupletsInGPU, ix, true);
                } else if (score_rphisum1 < score_rphisum2) {
                  rmQuintupletFromMemory(quintupletsInGPU, jx, true);
                } else {
                  rmQuintupletFromMemory(quintupletsInGPU, (ix < jx ? ix : jx), true);
                }
              }
            }
          }
        }
      }
    }
  };

  struct removeDupPixelTripletsInGPUFromMap {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, struct SDL::pixelTriplets pixelTripletsInGPU) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (unsigned int ix = globalThreadIdx[1]; ix < *pixelTripletsInGPU.nPixelTriplets; ix += gridThreadExtent[1]) {
        for (unsigned int jx = globalThreadIdx[2]; jx < *pixelTripletsInGPU.nPixelTriplets; jx += gridThreadExtent[2]) {
          if (ix == jx)
            continue;

          int nMatched[2];
          checkHitspT3(ix, jx, pixelTripletsInGPU, nMatched);
          if ((nMatched[0] + nMatched[1]) >= 5) {
            // Check the layers
            if (pixelTripletsInGPU.logicalLayers[5 * jx + 2] < pixelTripletsInGPU.logicalLayers[5 * ix + 2]) {
              rmPixelTripletFromMemory(pixelTripletsInGPU, ix);
              break;
            } else if (pixelTripletsInGPU.logicalLayers[5 * ix + 2] == pixelTripletsInGPU.logicalLayers[5 * jx + 2] &&
                       __H2F(pixelTripletsInGPU.score[ix]) > __H2F(pixelTripletsInGPU.score[jx])) {
              rmPixelTripletFromMemory(pixelTripletsInGPU, ix);
              break;
            } else if (pixelTripletsInGPU.logicalLayers[5 * ix + 2] == pixelTripletsInGPU.logicalLayers[5 * jx + 2] &&
                       (__H2F(pixelTripletsInGPU.score[ix]) == __H2F(pixelTripletsInGPU.score[jx])) && (ix < jx)) {
              rmPixelTripletFromMemory(pixelTripletsInGPU, ix);
              break;
            }
          }
        }
      }
    }
  };

  struct removeDupPixelQuintupletsInGPUFromMap {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, struct SDL::pixelQuintuplets pixelQuintupletsInGPU) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      unsigned int nPixelQuintuplets = *pixelQuintupletsInGPU.nPixelQuintuplets;
      for (unsigned int ix = globalThreadIdx[1]; ix < nPixelQuintuplets; ix += gridThreadExtent[1]) {
        float score1 = __H2F(pixelQuintupletsInGPU.score[ix]);
        for (unsigned int jx = globalThreadIdx[2]; jx < nPixelQuintuplets; jx += gridThreadExtent[2]) {
          if (ix == jx)
            continue;

          int nMatched = checkHitspT5(ix, jx, pixelQuintupletsInGPU);
          float score2 = __H2F(pixelQuintupletsInGPU.score[jx]);
          if (nMatched >= 7) {
            if (score1 > score2 or ((score1 == score2) and (ix > jx))) {
              rmPixelQuintupletFromMemory(pixelQuintupletsInGPU, ix);
              break;
            }
          }
        }
      }
    }
  };

  struct checkHitspLS {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  struct SDL::modules modulesInGPU,
                                  struct SDL::segments segmentsInGPU,
                                  bool secondpass) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      int pixelModuleIndex = *modulesInGPU.nLowerModules;
      unsigned int nPixelSegments = segmentsInGPU.nSegments[pixelModuleIndex];

      if (nPixelSegments > N_MAX_PIXEL_SEGMENTS_PER_MODULE)
        nPixelSegments = N_MAX_PIXEL_SEGMENTS_PER_MODULE;

      for (unsigned int ix = globalThreadIdx[1]; ix < nPixelSegments; ix += gridThreadExtent[1]) {
        if (secondpass && (!segmentsInGPU.isQuad[ix] || (segmentsInGPU.isDup[ix] & 1)))
          continue;

        unsigned int phits1[4];
        phits1[0] = segmentsInGPU.pLSHitsIdxs[ix].x;
        phits1[1] = segmentsInGPU.pLSHitsIdxs[ix].y;
        phits1[2] = segmentsInGPU.pLSHitsIdxs[ix].z;
        phits1[3] = segmentsInGPU.pLSHitsIdxs[ix].w;
        float eta_pix1 = segmentsInGPU.eta[ix];
        float phi_pix1 = segmentsInGPU.phi[ix];

        for (unsigned int jx = ix + 1 + globalThreadIdx[2]; jx < nPixelSegments; jx += gridThreadExtent[2]) {
          float eta_pix2 = segmentsInGPU.eta[jx];
          float phi_pix2 = segmentsInGPU.phi[jx];

          if (alpaka::math::abs(acc, eta_pix2 - eta_pix1) > 0.1f)
            continue;

          if (secondpass && (!segmentsInGPU.isQuad[jx] || (segmentsInGPU.isDup[jx] & 1)))
            continue;

          int8_t quad_diff = segmentsInGPU.isQuad[ix] - segmentsInGPU.isQuad[jx];
          float score_diff = segmentsInGPU.score[ix] - segmentsInGPU.score[jx];
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

          unsigned int phits2[4];
          phits2[0] = segmentsInGPU.pLSHitsIdxs[jx].x;
          phits2[1] = segmentsInGPU.pLSHitsIdxs[jx].y;
          phits2[2] = segmentsInGPU.pLSHitsIdxs[jx].z;
          phits2[3] = segmentsInGPU.pLSHitsIdxs[jx].w;

          int npMatched = 0;
          for (int i = 0; i < 4; i++) {
            bool pmatched = false;
            for (int j = 0; j < 4; j++) {
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
          if (npMatched >= 3) {
            rmPixelSegmentFromMemory(segmentsInGPU, idxToRemove, secondpass);
          }
          if (secondpass) {
            float dEta = alpaka::math::abs(acc, eta_pix1 - eta_pix2);
            float dPhi = SDL::calculate_dPhi(phi_pix1, phi_pix2);

            float dR2 = dEta * dEta + dPhi * dPhi;
            if ((npMatched >= 1) || (dR2 < 1e-5f)) {
              rmPixelSegmentFromMemory(segmentsInGPU, idxToRemove, secondpass);
            }
          }
        }
      }
    }
  };
}  // namespace SDL
#endif
