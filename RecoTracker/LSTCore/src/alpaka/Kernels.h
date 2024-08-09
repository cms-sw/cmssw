#ifndef RecoTracker_LSTCore_src_alpaka_Kernels_h
#define RecoTracker_LSTCore_src_alpaka_Kernels_h

#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#include "RecoTracker/LSTCore/interface/Module.h"

#include "Hit.h"
#include "MiniDoublet.h"
#include "ObjectRanges.h"
#include "Segment.h"
#include "Triplet.h"
#include "Quintuplet.h"
#include "PixelQuintuplet.h"
#include "PixelTriplet.h"

namespace lst {
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void rmQuintupletFromMemory(lst::Quintuplets& quintupletsInGPU,
                                                             unsigned int quintupletIndex,
                                                             bool secondpass = false) {
    quintupletsInGPU.isDup[quintupletIndex] |= 1 + secondpass;
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void rmPixelTripletFromMemory(lst::PixelTriplets& pixelTripletsInGPU,
                                                               unsigned int pixelTripletIndex) {
    pixelTripletsInGPU.isDup[pixelTripletIndex] = true;
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void rmPixelQuintupletFromMemory(lst::PixelQuintuplets& pixelQuintupletsInGPU,
                                                                  unsigned int pixelQuintupletIndex) {
    pixelQuintupletsInGPU.isDup[pixelQuintupletIndex] = true;
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void rmPixelSegmentFromMemory(lst::Segments& segmentsInGPU,
                                                               unsigned int pixelSegmentArrayIndex,
                                                               bool secondpass = false) {
    segmentsInGPU.isDup[pixelSegmentArrayIndex] |= 1 + secondpass;
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE int checkHitsT5(unsigned int ix,
                                                 unsigned int jx,
                                                 lst::Quintuplets const& quintupletsInGPU) {
    unsigned int hits1[Params_T5::kHits];
    unsigned int hits2[Params_T5::kHits];

    for (int i = 0; i < Params_T5::kHits; i++) {
      hits1[i] = quintupletsInGPU.hitIndices[Params_T5::kHits * ix + i];
      hits2[i] = quintupletsInGPU.hitIndices[Params_T5::kHits * jx + i];
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
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE int checkHitspT5(unsigned int ix,
                                                  unsigned int jx,
                                                  lst::PixelQuintuplets const& pixelQuintupletsInGPU) {
    unsigned int hits1[Params_pT5::kHits];
    unsigned int hits2[Params_pT5::kHits];

    for (int i = 0; i < Params_pT5::kHits; i++) {
      hits1[i] = pixelQuintupletsInGPU.hitIndices[Params_pT5::kHits * ix + i];
      hits2[i] = pixelQuintupletsInGPU.hitIndices[Params_pT5::kHits * jx + i];
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
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void checkHitspT3(unsigned int ix,
                                                   unsigned int jx,
                                                   lst::PixelTriplets const& pixelTripletsInGPU,
                                                   int* matched) {
    int phits1[Params_pLS::kHits];
    int phits2[Params_pLS::kHits];

    for (int i = 0; i < Params_pLS::kHits; i++) {
      phits1[i] = pixelTripletsInGPU.hitIndices[Params_pT3::kHits * ix + i];
      phits2[i] = pixelTripletsInGPU.hitIndices[Params_pT3::kHits * jx + i];
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
      hits1[i] = pixelTripletsInGPU.hitIndices[Params_pT3::kHits * ix + i + 4];  // Omitting the pLS hits
      hits2[i] = pixelTripletsInGPU.hitIndices[Params_pT3::kHits * jx + i + 4];  // Omitting the pLS hits
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
  };

  struct removeDupQuintupletsInGPUAfterBuild {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  lst::Modules modulesInGPU,
                                  lst::Quintuplets quintupletsInGPU,
                                  lst::ObjectRanges rangesInGPU) const {
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
            float dPhi = lst::calculate_dPhi(phi1, phi2);
            float score_rphisum2 = __H2F(quintupletsInGPU.score_rphisum[jx]);

            if (dEta > 0.1f)
              continue;

            if (alpaka::math::abs(acc, dPhi) > 0.1f)
              continue;

            int nMatched = checkHitsT5(ix, jx, quintupletsInGPU);
            const int minNHitsForDup_T5 = 7;
            if (nMatched >= minNHitsForDup_T5) {
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
                                  lst::Quintuplets quintupletsInGPU,
                                  lst::ObjectRanges rangesInGPU) const {
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
              float dPhi = lst::calculate_dPhi(phi1, phi2);

              if (dEta > 0.1f)
                continue;

              if (alpaka::math::abs(acc, dPhi) > 0.1f)
                continue;

              float dR2 = dEta * dEta + dPhi * dPhi;
              int nMatched = checkHitsT5(ix, jx, quintupletsInGPU);
              const int minNHitsForDup_T5 = 5;
              if (dR2 < 0.001f || nMatched >= minNHitsForDup_T5) {
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
    ALPAKA_FN_ACC void operator()(TAcc const& acc, lst::PixelTriplets pixelTripletsInGPU) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (unsigned int ix = globalThreadIdx[1]; ix < *pixelTripletsInGPU.nPixelTriplets; ix += gridThreadExtent[1]) {
        for (unsigned int jx = globalThreadIdx[2]; jx < *pixelTripletsInGPU.nPixelTriplets; jx += gridThreadExtent[2]) {
          if (ix == jx)
            continue;

          int nMatched[2];
          checkHitspT3(ix, jx, pixelTripletsInGPU, nMatched);
          const int minNHitsForDup_pT3 = 5;
          if ((nMatched[0] + nMatched[1]) >= minNHitsForDup_pT3) {
            // Check the layers
            if (pixelTripletsInGPU.logicalLayers[Params_pT3::kLayers * jx + 2] <
                pixelTripletsInGPU.logicalLayers[Params_pT3::kLayers * ix + 2]) {
              rmPixelTripletFromMemory(pixelTripletsInGPU, ix);
              break;
            } else if (pixelTripletsInGPU.logicalLayers[Params_pT3::kLayers * ix + 2] ==
                           pixelTripletsInGPU.logicalLayers[Params_pT3::kLayers * jx + 2] &&
                       __H2F(pixelTripletsInGPU.score[ix]) > __H2F(pixelTripletsInGPU.score[jx])) {
              rmPixelTripletFromMemory(pixelTripletsInGPU, ix);
              break;
            } else if (pixelTripletsInGPU.logicalLayers[Params_pT3::kLayers * ix + 2] ==
                           pixelTripletsInGPU.logicalLayers[Params_pT3::kLayers * jx + 2] &&
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
    ALPAKA_FN_ACC void operator()(TAcc const& acc, lst::PixelQuintuplets pixelQuintupletsInGPU) const {
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
          const int minNHitsForDup_pT5 = 7;
          if (nMatched >= minNHitsForDup_pT5) {
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
                                  lst::Modules modulesInGPU,
                                  lst::Segments segmentsInGPU,
                                  bool secondpass) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      int pixelModuleIndex = *modulesInGPU.nLowerModules;
      unsigned int nPixelSegments = segmentsInGPU.nSegments[pixelModuleIndex];

      if (nPixelSegments > n_max_pixel_segments_per_module)
        nPixelSegments = n_max_pixel_segments_per_module;

      for (unsigned int ix = globalThreadIdx[1]; ix < nPixelSegments; ix += gridThreadExtent[1]) {
        if (secondpass && (!segmentsInGPU.isQuad[ix] || (segmentsInGPU.isDup[ix] & 1)))
          continue;

        unsigned int phits1[Params_pLS::kHits];
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

          unsigned int phits2[Params_pLS::kHits];
          phits2[0] = segmentsInGPU.pLSHitsIdxs[jx].x;
          phits2[1] = segmentsInGPU.pLSHitsIdxs[jx].y;
          phits2[2] = segmentsInGPU.pLSHitsIdxs[jx].z;
          phits2[3] = segmentsInGPU.pLSHitsIdxs[jx].w;

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
            rmPixelSegmentFromMemory(segmentsInGPU, idxToRemove, secondpass);
          }
          if (secondpass) {
            float dEta = alpaka::math::abs(acc, eta_pix1 - eta_pix2);
            float dPhi = lst::calculate_dPhi(phi_pix1, phi_pix2);

            float dR2 = dEta * dEta + dPhi * dPhi;
            if ((npMatched >= 1) || (dR2 < 1e-5f)) {
              rmPixelSegmentFromMemory(segmentsInGPU, idxToRemove, secondpass);
            }
          }
        }
      }
    }
  };
}  // namespace lst
#endif
