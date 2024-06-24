#ifndef TrackCandidate_cuh
#define TrackCandidate_cuh

#ifdef LST_IS_CMSSW_PACKAGE
#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#include "RecoTracker/LSTCore/interface/alpaka/Module.h"
#else
#include "Constants.h"
#include "Module.h"
#endif

#include "Triplet.h"
#include "Segment.h"
#include "MiniDoublet.h"
#include "PixelTriplet.h"
#include "Quintuplet.h"
#include "Hit.h"

namespace SDL {
  struct trackCandidates {
    short* trackCandidateType;          // 4-T5 5-pT3 7-pT5 8-pLS
    unsigned int* directObjectIndices;  // Will hold direct indices to each type containers
    unsigned int* objectIndices;        // Will hold tracklet and  triplet indices - check the type!!
    unsigned int* nTrackCandidates;
    unsigned int* nTrackCandidatespT3;
    unsigned int* nTrackCandidatespT5;
    unsigned int* nTrackCandidatespLS;
    unsigned int* nTrackCandidatesT5;

    uint8_t* logicalLayers;
    unsigned int* hitIndices;
    int* pixelSeedIndex;
    uint16_t* lowerModuleIndices;

    FPX* centerX;
    FPX* centerY;
    FPX* radius;

    template <typename TBuff>
    void setData(TBuff& trackCandidatesbuf) {
      trackCandidateType = alpaka::getPtrNative(trackCandidatesbuf.trackCandidateType_buf);
      directObjectIndices = alpaka::getPtrNative(trackCandidatesbuf.directObjectIndices_buf);
      objectIndices = alpaka::getPtrNative(trackCandidatesbuf.objectIndices_buf);
      nTrackCandidates = alpaka::getPtrNative(trackCandidatesbuf.nTrackCandidates_buf);
      nTrackCandidatespT3 = alpaka::getPtrNative(trackCandidatesbuf.nTrackCandidatespT3_buf);
      nTrackCandidatespT5 = alpaka::getPtrNative(trackCandidatesbuf.nTrackCandidatespT5_buf);
      nTrackCandidatespLS = alpaka::getPtrNative(trackCandidatesbuf.nTrackCandidatespLS_buf);
      nTrackCandidatesT5 = alpaka::getPtrNative(trackCandidatesbuf.nTrackCandidatesT5_buf);

      logicalLayers = alpaka::getPtrNative(trackCandidatesbuf.logicalLayers_buf);
      hitIndices = alpaka::getPtrNative(trackCandidatesbuf.hitIndices_buf);
      pixelSeedIndex = alpaka::getPtrNative(trackCandidatesbuf.pixelSeedIndex_buf);
      lowerModuleIndices = alpaka::getPtrNative(trackCandidatesbuf.lowerModuleIndices_buf);

      centerX = alpaka::getPtrNative(trackCandidatesbuf.centerX_buf);
      centerY = alpaka::getPtrNative(trackCandidatesbuf.centerY_buf);
      radius = alpaka::getPtrNative(trackCandidatesbuf.radius_buf);
    }
  };

  template <typename TDev>
  struct trackCandidatesBuffer : trackCandidates {
    Buf<TDev, short> trackCandidateType_buf;
    Buf<TDev, unsigned int> directObjectIndices_buf;
    Buf<TDev, unsigned int> objectIndices_buf;
    Buf<TDev, unsigned int> nTrackCandidates_buf;
    Buf<TDev, unsigned int> nTrackCandidatespT3_buf;
    Buf<TDev, unsigned int> nTrackCandidatespT5_buf;
    Buf<TDev, unsigned int> nTrackCandidatespLS_buf;
    Buf<TDev, unsigned int> nTrackCandidatesT5_buf;

    Buf<TDev, uint8_t> logicalLayers_buf;
    Buf<TDev, unsigned int> hitIndices_buf;
    Buf<TDev, int> pixelSeedIndex_buf;
    Buf<TDev, uint16_t> lowerModuleIndices_buf;

    Buf<TDev, FPX> centerX_buf;
    Buf<TDev, FPX> centerY_buf;
    Buf<TDev, FPX> radius_buf;

    template <typename TQueue, typename TDevAcc>
    trackCandidatesBuffer(unsigned int maxTrackCandidates, TDevAcc const& devAccIn, TQueue& queue)
        : trackCandidateType_buf(allocBufWrapper<short>(devAccIn, maxTrackCandidates, queue)),
          directObjectIndices_buf(allocBufWrapper<unsigned int>(devAccIn, maxTrackCandidates, queue)),
          objectIndices_buf(allocBufWrapper<unsigned int>(devAccIn, 2 * maxTrackCandidates, queue)),
          nTrackCandidates_buf(allocBufWrapper<unsigned int>(devAccIn, 1, queue)),
          nTrackCandidatespT3_buf(allocBufWrapper<unsigned int>(devAccIn, 1, queue)),
          nTrackCandidatespT5_buf(allocBufWrapper<unsigned int>(devAccIn, 1, queue)),
          nTrackCandidatespLS_buf(allocBufWrapper<unsigned int>(devAccIn, 1, queue)),
          nTrackCandidatesT5_buf(allocBufWrapper<unsigned int>(devAccIn, 1, queue)),
          logicalLayers_buf(allocBufWrapper<uint8_t>(devAccIn, 7 * maxTrackCandidates, queue)),
          hitIndices_buf(allocBufWrapper<unsigned int>(devAccIn, 14 * maxTrackCandidates, queue)),
          pixelSeedIndex_buf(allocBufWrapper<int>(devAccIn, maxTrackCandidates, queue)),
          lowerModuleIndices_buf(allocBufWrapper<uint16_t>(devAccIn, 7 * maxTrackCandidates, queue)),
          centerX_buf(allocBufWrapper<FPX>(devAccIn, maxTrackCandidates, queue)),
          centerY_buf(allocBufWrapper<FPX>(devAccIn, maxTrackCandidates, queue)),
          radius_buf(allocBufWrapper<FPX>(devAccIn, maxTrackCandidates, queue)) {
      alpaka::memset(queue, nTrackCandidates_buf, 0u);
      alpaka::memset(queue, nTrackCandidatesT5_buf, 0u);
      alpaka::memset(queue, nTrackCandidatespT3_buf, 0u);
      alpaka::memset(queue, nTrackCandidatespT5_buf, 0u);
      alpaka::memset(queue, nTrackCandidatespLS_buf, 0u);
      alpaka::memset(queue, logicalLayers_buf, 0u);
      alpaka::memset(queue, lowerModuleIndices_buf, 0u);
      alpaka::memset(queue, hitIndices_buf, 0u);
      alpaka::memset(queue, pixelSeedIndex_buf, 0);
      alpaka::wait(queue);
    }
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addpLSTrackCandidateToMemory(struct SDL::trackCandidates& trackCandidatesInGPU,
                                                                   unsigned int trackletIndex,
                                                                   unsigned int trackCandidateIndex,
                                                                   uint4 hitIndices,
                                                                   int pixelSeedIndex) {
    trackCandidatesInGPU.trackCandidateType[trackCandidateIndex] = 8;
    trackCandidatesInGPU.directObjectIndices[trackCandidateIndex] = trackletIndex;
    trackCandidatesInGPU.pixelSeedIndex[trackCandidateIndex] = pixelSeedIndex;

    trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex] = trackletIndex;
    trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex + 1] = trackletIndex;

    trackCandidatesInGPU.hitIndices[14 * trackCandidateIndex + 0] =
        hitIndices.x;  // Order explanation in https://github.com/SegmentLinking/TrackLooper/issues/267
    trackCandidatesInGPU.hitIndices[14 * trackCandidateIndex + 1] = hitIndices.z;
    trackCandidatesInGPU.hitIndices[14 * trackCandidateIndex + 2] = hitIndices.y;
    trackCandidatesInGPU.hitIndices[14 * trackCandidateIndex + 3] = hitIndices.w;
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addTrackCandidateToMemory(struct SDL::trackCandidates& trackCandidatesInGPU,
                                                                short trackCandidateType,
                                                                unsigned int innerTrackletIndex,
                                                                unsigned int outerTrackletIndex,
                                                                uint8_t* logicalLayerIndices,
                                                                uint16_t* lowerModuleIndices,
                                                                unsigned int* hitIndices,
                                                                int pixelSeedIndex,
                                                                float centerX,
                                                                float centerY,
                                                                float radius,
                                                                unsigned int trackCandidateIndex,
                                                                unsigned int directObjectIndex) {
    trackCandidatesInGPU.trackCandidateType[trackCandidateIndex] = trackCandidateType;
    trackCandidatesInGPU.directObjectIndices[trackCandidateIndex] = directObjectIndex;
    trackCandidatesInGPU.pixelSeedIndex[trackCandidateIndex] = pixelSeedIndex;

    trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex] = innerTrackletIndex;
    trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex + 1] = outerTrackletIndex;

    size_t limits = trackCandidateType == 7 ? 7 : 5;

    //send the starting pointer to the logicalLayer and hitIndices
    for (size_t i = 0; i < limits; i++) {
      trackCandidatesInGPU.logicalLayers[7 * trackCandidateIndex + i] = logicalLayerIndices[i];
      trackCandidatesInGPU.lowerModuleIndices[7 * trackCandidateIndex + i] = lowerModuleIndices[i];
    }
    for (size_t i = 0; i < 2 * limits; i++) {
      trackCandidatesInGPU.hitIndices[14 * trackCandidateIndex + i] = hitIndices[i];
    }
    trackCandidatesInGPU.centerX[trackCandidateIndex] = __F2H(centerX);
    trackCandidatesInGPU.centerY[trackCandidateIndex] = __F2H(centerY);
    trackCandidatesInGPU.radius[trackCandidateIndex] = __F2H(radius);
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE int checkPixelHits(unsigned int ix,
                                                    unsigned int jx,
                                                    struct SDL::miniDoublets& mdsInGPU,
                                                    struct SDL::segments& segmentsInGPU,
                                                    struct SDL::hits& hitsInGPU) {
    int phits1[4] = {-1, -1, -1, -1};
    int phits2[4] = {-1, -1, -1, -1};
    phits1[0] = hitsInGPU.idxs[mdsInGPU.anchorHitIndices[segmentsInGPU.mdIndices[2 * ix]]];
    phits1[1] = hitsInGPU.idxs[mdsInGPU.anchorHitIndices[segmentsInGPU.mdIndices[2 * ix + 1]]];
    phits1[2] = hitsInGPU.idxs[mdsInGPU.outerHitIndices[segmentsInGPU.mdIndices[2 * ix]]];
    phits1[3] = hitsInGPU.idxs[mdsInGPU.outerHitIndices[segmentsInGPU.mdIndices[2 * ix + 1]]];

    phits2[0] = hitsInGPU.idxs[mdsInGPU.anchorHitIndices[segmentsInGPU.mdIndices[2 * jx]]];
    phits2[1] = hitsInGPU.idxs[mdsInGPU.anchorHitIndices[segmentsInGPU.mdIndices[2 * jx + 1]]];
    phits2[2] = hitsInGPU.idxs[mdsInGPU.outerHitIndices[segmentsInGPU.mdIndices[2 * jx]]];
    phits2[3] = hitsInGPU.idxs[mdsInGPU.outerHitIndices[segmentsInGPU.mdIndices[2 * jx + 1]]];

    int npMatched = 0;

    for (int i = 0; i < 4; i++) {
      bool pmatched = false;
      if (phits1[i] == -1)
        continue;

      for (int j = 0; j < 4; j++) {
        if (phits2[j] == -1)
          continue;

        if (phits1[i] == phits2[j]) {
          pmatched = true;
          break;
        }
      }
      if (pmatched)
        npMatched++;
    }
    return npMatched;
  };

  struct crossCleanpT3 {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  struct SDL::modules modulesInGPU,
                                  struct SDL::objectRanges rangesInGPU,
                                  struct SDL::pixelTriplets pixelTripletsInGPU,
                                  struct SDL::segments segmentsInGPU,
                                  struct SDL::pixelQuintuplets pixelQuintupletsInGPU) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      unsigned int nPixelTriplets = *pixelTripletsInGPU.nPixelTriplets;
      for (unsigned int pixelTripletIndex = globalThreadIdx[2]; pixelTripletIndex < nPixelTriplets;
           pixelTripletIndex += gridThreadExtent[2]) {
        if (pixelTripletsInGPU.isDup[pixelTripletIndex])
          continue;

        // Cross cleaning step
        float eta1 = __H2F(pixelTripletsInGPU.eta_pix[pixelTripletIndex]);
        float phi1 = __H2F(pixelTripletsInGPU.phi_pix[pixelTripletIndex]);

        int pixelModuleIndex = *modulesInGPU.nLowerModules;
        unsigned int prefix = rangesInGPU.segmentModuleIndices[pixelModuleIndex];

        unsigned int nPixelQuintuplets = *pixelQuintupletsInGPU.nPixelQuintuplets;
        for (unsigned int pixelQuintupletIndex = globalThreadIdx[1]; pixelQuintupletIndex < nPixelQuintuplets;
             pixelQuintupletIndex += gridThreadExtent[1]) {
          unsigned int pLS_jx = pixelQuintupletsInGPU.pixelIndices[pixelQuintupletIndex];
          float eta2 = segmentsInGPU.eta[pLS_jx - prefix];
          float phi2 = segmentsInGPU.phi[pLS_jx - prefix];
          float dEta = alpaka::math::abs(acc, (eta1 - eta2));
          float dPhi = SDL::calculate_dPhi(phi1, phi2);

          float dR2 = dEta * dEta + dPhi * dPhi;
          if (dR2 < 1e-5f)
            pixelTripletsInGPU.isDup[pixelTripletIndex] = true;
        }
      }
    }
  };

  struct crossCleanT5 {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  struct SDL::modules modulesInGPU,
                                  struct SDL::quintuplets quintupletsInGPU,
                                  struct SDL::pixelQuintuplets pixelQuintupletsInGPU,
                                  struct SDL::pixelTriplets pixelTripletsInGPU,
                                  struct SDL::objectRanges rangesInGPU) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (int innerInnerInnerLowerModuleArrayIndex = globalThreadIdx[0];
           innerInnerInnerLowerModuleArrayIndex < *(modulesInGPU.nLowerModules);
           innerInnerInnerLowerModuleArrayIndex += gridThreadExtent[0]) {
        if (rangesInGPU.quintupletModuleIndices[innerInnerInnerLowerModuleArrayIndex] == -1)
          continue;

        unsigned int nQuints = quintupletsInGPU.nQuintuplets[innerInnerInnerLowerModuleArrayIndex];
        for (unsigned int innerObjectArrayIndex = globalThreadIdx[1]; innerObjectArrayIndex < nQuints;
             innerObjectArrayIndex += gridThreadExtent[1]) {
          unsigned int quintupletIndex =
              rangesInGPU.quintupletModuleIndices[innerInnerInnerLowerModuleArrayIndex] + innerObjectArrayIndex;

          // Don't add duplicate T5s or T5s that are accounted in pT5s
          if (quintupletsInGPU.isDup[quintupletIndex] or quintupletsInGPU.partOfPT5[quintupletIndex])
            continue;
#ifdef Crossclean_T5
          unsigned int loop_bound = *pixelQuintupletsInGPU.nPixelQuintuplets + *pixelTripletsInGPU.nPixelTriplets;
          // Cross cleaning step
          float eta1 = __H2F(quintupletsInGPU.eta[quintupletIndex]);
          float phi1 = __H2F(quintupletsInGPU.phi[quintupletIndex]);

          for (unsigned int jx = globalThreadIdx[2]; jx < loop_bound; jx += gridThreadExtent[2]) {
            float eta2, phi2;
            if (jx < *pixelQuintupletsInGPU.nPixelQuintuplets) {
              eta2 = __H2F(pixelQuintupletsInGPU.eta[jx]);
              phi2 = __H2F(pixelQuintupletsInGPU.phi[jx]);
            } else {
              eta2 = __H2F(pixelTripletsInGPU.eta[jx - *pixelQuintupletsInGPU.nPixelQuintuplets]);
              phi2 = __H2F(pixelTripletsInGPU.phi[jx - *pixelQuintupletsInGPU.nPixelQuintuplets]);
            }

            float dEta = alpaka::math::abs(acc, eta1 - eta2);
            float dPhi = SDL::calculate_dPhi(phi1, phi2);

            float dR2 = dEta * dEta + dPhi * dPhi;
            if (dR2 < 1e-3f)
              quintupletsInGPU.isDup[quintupletIndex] = true;
          }
#endif
        }
      }
    }
  };

  // Using Matt's block for the outer loop and thread for inner loop trick here!
  // This will eliminate the need for another kernel just for adding the pLS, because we can __syncthreads()
  struct crossCleanpLS {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  struct SDL::modules modulesInGPU,
                                  struct SDL::objectRanges rangesInGPU,
                                  struct SDL::pixelTriplets pixelTripletsInGPU,
                                  struct SDL::trackCandidates trackCandidatesInGPU,
                                  struct SDL::segments segmentsInGPU,
                                  struct SDL::miniDoublets mdsInGPU,
                                  struct SDL::hits hitsInGPU,
                                  struct SDL::quintuplets quintupletsInGPU) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      int pixelModuleIndex = *modulesInGPU.nLowerModules;
      unsigned int nPixels = segmentsInGPU.nSegments[pixelModuleIndex];
      for (unsigned int pixelArrayIndex = globalThreadIdx[2]; pixelArrayIndex < nPixels;
           pixelArrayIndex += gridThreadExtent[2]) {
        if (!segmentsInGPU.isQuad[pixelArrayIndex] || segmentsInGPU.isDup[pixelArrayIndex])
          continue;

        float eta1 = segmentsInGPU.eta[pixelArrayIndex];
        float phi1 = segmentsInGPU.phi[pixelArrayIndex];
        unsigned int prefix = rangesInGPU.segmentModuleIndices[pixelModuleIndex];

        unsigned int nTrackCandidates = *(trackCandidatesInGPU.nTrackCandidates);
        for (unsigned int trackCandidateIndex = globalThreadIdx[1]; trackCandidateIndex < nTrackCandidates;
             trackCandidateIndex += gridThreadExtent[1]) {
          short type = trackCandidatesInGPU.trackCandidateType[trackCandidateIndex];
          unsigned int innerTrackletIdx = trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex];
          if (type == 4)  // T5
          {
            unsigned int quintupletIndex = innerTrackletIdx;  // T5 index
            float eta2 = __H2F(quintupletsInGPU.eta[quintupletIndex]);
            float phi2 = __H2F(quintupletsInGPU.phi[quintupletIndex]);
            float dEta = alpaka::math::abs(acc, eta1 - eta2);
            float dPhi = SDL::calculate_dPhi(phi1, phi2);

            float dR2 = dEta * dEta + dPhi * dPhi;
            if (dR2 < 1e-3f)
              segmentsInGPU.isDup[pixelArrayIndex] = true;
          }
          if (type == 5)  // pT3
          {
            int pLSIndex = pixelTripletsInGPU.pixelSegmentIndices[innerTrackletIdx];
            int npMatched = checkPixelHits(prefix + pixelArrayIndex, pLSIndex, mdsInGPU, segmentsInGPU, hitsInGPU);
            if (npMatched > 0)
              segmentsInGPU.isDup[pixelArrayIndex] = true;

            int pT3Index = innerTrackletIdx;
            float eta2 = __H2F(pixelTripletsInGPU.eta_pix[pT3Index]);
            float phi2 = __H2F(pixelTripletsInGPU.phi_pix[pT3Index]);
            float dEta = alpaka::math::abs(acc, eta1 - eta2);
            float dPhi = SDL::calculate_dPhi(phi1, phi2);

            float dR2 = dEta * dEta + dPhi * dPhi;
            if (dR2 < 0.000001f)
              segmentsInGPU.isDup[pixelArrayIndex] = true;
          }
          if (type == 7)  // pT5
          {
            unsigned int pLSIndex = innerTrackletIdx;
            int npMatched = checkPixelHits(prefix + pixelArrayIndex, pLSIndex, mdsInGPU, segmentsInGPU, hitsInGPU);
            if (npMatched > 0) {
              segmentsInGPU.isDup[pixelArrayIndex] = true;
            }

            float eta2 = segmentsInGPU.eta[pLSIndex - prefix];
            float phi2 = segmentsInGPU.phi[pLSIndex - prefix];
            float dEta = alpaka::math::abs(acc, eta1 - eta2);
            float dPhi = SDL::calculate_dPhi(phi1, phi2);

            float dR2 = dEta * dEta + dPhi * dPhi;
            if (dR2 < 0.000001f)
              segmentsInGPU.isDup[pixelArrayIndex] = true;
          }
        }
      }
    }
  };

  struct addpT3asTrackCandidatesInGPU {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  uint16_t nLowerModules,
                                  struct SDL::pixelTriplets pixelTripletsInGPU,
                                  struct SDL::trackCandidates trackCandidatesInGPU,
                                  struct SDL::segments segmentsInGPU,
                                  struct SDL::objectRanges rangesInGPU) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      unsigned int nPixelTriplets = *pixelTripletsInGPU.nPixelTriplets;
      unsigned int pLS_offset = rangesInGPU.segmentModuleIndices[nLowerModules];
      for (unsigned int pixelTripletIndex = globalThreadIdx[2]; pixelTripletIndex < nPixelTriplets;
           pixelTripletIndex += gridThreadExtent[2]) {
        if ((pixelTripletsInGPU.isDup[pixelTripletIndex]))
          continue;

        unsigned int trackCandidateIdx =
            alpaka::atomicOp<alpaka::AtomicAdd>(acc, trackCandidatesInGPU.nTrackCandidates, 1u);
        if (trackCandidateIdx >= N_MAX_PIXEL_TRACK_CANDIDATES)  // This is done before any non-pixel TCs are added
        {
#ifdef Warnings
          printf("Track Candidate excess alert! Type = pT3");
#endif
          alpaka::atomicOp<alpaka::AtomicSub>(acc, trackCandidatesInGPU.nTrackCandidates, 1u);
          break;

        } else {
          alpaka::atomicOp<alpaka::AtomicAdd>(acc, trackCandidatesInGPU.nTrackCandidatespT3, 1u);

          float radius = 0.5f * (__H2F(pixelTripletsInGPU.pixelRadius[pixelTripletIndex]) +
                                 __H2F(pixelTripletsInGPU.tripletRadius[pixelTripletIndex]));
          unsigned int pT3PixelIndex = pixelTripletsInGPU.pixelSegmentIndices[pixelTripletIndex];
          addTrackCandidateToMemory(trackCandidatesInGPU,
                                    5 /*track candidate type pT3=5*/,
                                    pixelTripletIndex,
                                    pixelTripletIndex,
                                    &pixelTripletsInGPU.logicalLayers[5 * pixelTripletIndex],
                                    &pixelTripletsInGPU.lowerModuleIndices[5 * pixelTripletIndex],
                                    &pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex],
                                    segmentsInGPU.seedIdx[pT3PixelIndex - pLS_offset],
                                    __H2F(pixelTripletsInGPU.centerX[pixelTripletIndex]),
                                    __H2F(pixelTripletsInGPU.centerY[pixelTripletIndex]),
                                    radius,
                                    trackCandidateIdx,
                                    pixelTripletIndex);
        }
      }
    }
  };

  struct addT5asTrackCandidateInGPU {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  uint16_t nLowerModules,
                                  struct SDL::quintuplets quintupletsInGPU,
                                  struct SDL::trackCandidates trackCandidatesInGPU,
                                  struct SDL::objectRanges rangesInGPU) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (int idx = globalThreadIdx[1]; idx < nLowerModules; idx += gridThreadExtent[1]) {
        if (rangesInGPU.quintupletModuleIndices[idx] == -1)
          continue;

        unsigned int nQuints = quintupletsInGPU.nQuintuplets[idx];
        for (unsigned int jdx = globalThreadIdx[2]; jdx < nQuints; jdx += gridThreadExtent[2]) {
          unsigned int quintupletIndex = rangesInGPU.quintupletModuleIndices[idx] + jdx;
          if (quintupletsInGPU.isDup[quintupletIndex] or quintupletsInGPU.partOfPT5[quintupletIndex])
            continue;
          if (!(quintupletsInGPU.TightCutFlag[quintupletIndex]))
            continue;

          unsigned int trackCandidateIdx =
              alpaka::atomicOp<alpaka::AtomicAdd>(acc, trackCandidatesInGPU.nTrackCandidates, 1u);
          if (trackCandidateIdx - *trackCandidatesInGPU.nTrackCandidatespT5 -
                  *trackCandidatesInGPU.nTrackCandidatespT3 >=
              N_MAX_NONPIXEL_TRACK_CANDIDATES)  // pT5 and pT3 TCs have been added, but not pLS TCs
          {
#ifdef Warnings
            printf("Track Candidate excess alert! Type = T5");
#endif
            alpaka::atomicOp<alpaka::AtomicSub>(acc, trackCandidatesInGPU.nTrackCandidates, 1u);
            break;
          } else {
            alpaka::atomicOp<alpaka::AtomicAdd>(acc, trackCandidatesInGPU.nTrackCandidatesT5, 1u);
            addTrackCandidateToMemory(trackCandidatesInGPU,
                                      4 /*track candidate type T5=4*/,
                                      quintupletIndex,
                                      quintupletIndex,
                                      &quintupletsInGPU.logicalLayers[5 * quintupletIndex],
                                      &quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex],
                                      &quintupletsInGPU.hitIndices[10 * quintupletIndex],
                                      -1 /*no pixel seed index for T5s*/,
                                      quintupletsInGPU.regressionG[quintupletIndex],
                                      quintupletsInGPU.regressionF[quintupletIndex],
                                      quintupletsInGPU.regressionRadius[quintupletIndex],
                                      trackCandidateIdx,
                                      quintupletIndex);
          }
        }
      }
    }
  };

  struct addpLSasTrackCandidateInGPU {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  uint16_t nLowerModules,
                                  struct SDL::trackCandidates trackCandidatesInGPU,
                                  struct SDL::segments segmentsInGPU) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      unsigned int nPixels = segmentsInGPU.nSegments[nLowerModules];
      for (unsigned int pixelArrayIndex = globalThreadIdx[2]; pixelArrayIndex < nPixels;
           pixelArrayIndex += gridThreadExtent[2]) {
#ifdef TC_PLS_TRIPLETS
        if (segmentsInGPU.isDup[pixelArrayIndex])
#else
        if ((!segmentsInGPU.isQuad[pixelArrayIndex]) || (segmentsInGPU.isDup[pixelArrayIndex]))
#endif
          continue;

        unsigned int trackCandidateIdx =
            alpaka::atomicOp<alpaka::AtomicAdd>(acc, trackCandidatesInGPU.nTrackCandidates, 1u);
        if (trackCandidateIdx - *trackCandidatesInGPU.nTrackCandidatesT5 >=
            N_MAX_PIXEL_TRACK_CANDIDATES)  // T5 TCs have already been added
        {
#ifdef Warnings
          printf("Track Candidate excess alert! Type = pLS");
#endif
          alpaka::atomicOp<alpaka::AtomicSub>(acc, trackCandidatesInGPU.nTrackCandidates, 1u);
          break;

        } else {
          alpaka::atomicOp<alpaka::AtomicAdd>(acc, trackCandidatesInGPU.nTrackCandidatespLS, 1u);
          addpLSTrackCandidateToMemory(trackCandidatesInGPU,
                                       pixelArrayIndex,
                                       trackCandidateIdx,
                                       segmentsInGPU.pLSHitsIdxs[pixelArrayIndex],
                                       segmentsInGPU.seedIdx[pixelArrayIndex]);
        }
      }
    }
  };

  struct addpT5asTrackCandidateInGPU {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  uint16_t nLowerModules,
                                  struct SDL::pixelQuintuplets pixelQuintupletsInGPU,
                                  struct SDL::trackCandidates trackCandidatesInGPU,
                                  struct SDL::segments segmentsInGPU,
                                  struct SDL::objectRanges rangesInGPU) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      int nPixelQuintuplets = *pixelQuintupletsInGPU.nPixelQuintuplets;
      unsigned int pLS_offset = rangesInGPU.segmentModuleIndices[nLowerModules];
      for (int pixelQuintupletIndex = globalThreadIdx[2]; pixelQuintupletIndex < nPixelQuintuplets;
           pixelQuintupletIndex += gridThreadExtent[2]) {
        if (pixelQuintupletsInGPU.isDup[pixelQuintupletIndex])
          continue;

        unsigned int trackCandidateIdx =
            alpaka::atomicOp<alpaka::AtomicAdd>(acc, trackCandidatesInGPU.nTrackCandidates, 1u);
        if (trackCandidateIdx >= N_MAX_PIXEL_TRACK_CANDIDATES)  // No other TCs have been added yet
        {
#ifdef Warnings
          printf("Track Candidate excess alert! Type = pT5");
#endif
          alpaka::atomicOp<alpaka::AtomicSub>(acc, trackCandidatesInGPU.nTrackCandidates, 1u);
          break;

        } else {
          alpaka::atomicOp<alpaka::AtomicAdd>(acc, trackCandidatesInGPU.nTrackCandidatespT5, 1u);

          float radius = 0.5f * (__H2F(pixelQuintupletsInGPU.pixelRadius[pixelQuintupletIndex]) +
                                 __H2F(pixelQuintupletsInGPU.quintupletRadius[pixelQuintupletIndex]));
          unsigned int pT5PixelIndex = pixelQuintupletsInGPU.pixelIndices[pixelQuintupletIndex];
          addTrackCandidateToMemory(trackCandidatesInGPU,
                                    7 /*track candidate type pT5=7*/,
                                    pT5PixelIndex,
                                    pixelQuintupletsInGPU.T5Indices[pixelQuintupletIndex],
                                    &pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex],
                                    &pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex],
                                    &pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex],
                                    segmentsInGPU.seedIdx[pT5PixelIndex - pLS_offset],
                                    __H2F(pixelQuintupletsInGPU.centerX[pixelQuintupletIndex]),
                                    __H2F(pixelQuintupletsInGPU.centerY[pixelQuintupletIndex]),
                                    radius,
                                    trackCandidateIdx,
                                    pixelQuintupletIndex);
        }
      }
    }
  };
}  // namespace SDL
#endif
