#include "RecoTracker/LSTCore/interface/alpaka/LST.h"

#include "LSTEvent.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE::lst;

#include "Math/Vector3D.h"
#include "Math/VectorUtil.h"
using XYZVector = ROOT::Math::XYZVector;

namespace {
  using namespace ALPAKA_ACCELERATOR_NAMESPACE::lst;
  std::vector<unsigned int> getHitIdxs(short trackCandidateType,
                                       Params_pT5::ArrayUxHits const& tcHitIndices,
                                       unsigned int const* hitIndices) {
    std::vector<unsigned int> hits;

    unsigned int maxNHits = 0;
    if (trackCandidateType == LSTObjType::pT5)
      maxNHits = Params_pT5::kHits;
    else if (trackCandidateType == LSTObjType::pT3)
      maxNHits = Params_pT3::kHits;
    else if (trackCandidateType == LSTObjType::T5)
      maxNHits = Params_T5::kHits;
    else if (trackCandidateType == LSTObjType::pLS)
      maxNHits = Params_pLS::kHits;

    for (unsigned int i = 0; i < maxNHits; i++) {
      unsigned int hitIdxDev = tcHitIndices[i];
      unsigned int hitIdx =
          (trackCandidateType == LSTObjType::pLS)
              ? hitIdxDev
              : hitIndices[hitIdxDev];  // Hit indices are stored differently in the standalone for pLS.

      // For p objects, the 3rd and 4th hit maybe the same,
      // due to the way pLS hits are stored in the standalone.
      // This is because pixel seeds can be either triplets or quadruplets.
      if (trackCandidateType != LSTObjType::T5 && hits.size() == 3 &&
          hits.back() == hitIdx)  // Remove duplicate 4th hits.
        continue;

      hits.push_back(hitIdx);
    }

    return hits;
  }

}  // namespace

void LST::getOutput(LSTEvent& event) {
  out_tc_hitIdxs_.clear();
  out_tc_len_.clear();
  out_tc_seedIdx_.clear();
  out_tc_trackCandidateType_.clear();

  auto const hits = event.getHits<HitsSoA>(/*inCMSSW*/ true, /*sync*/ false);  // sync on next line
  auto const& trackCandidates = event.getTrackCandidates(/*inCMSSW*/ true, /*sync*/ true);

  unsigned int nTrackCandidates = trackCandidates.nTrackCandidates();

  for (unsigned int idx = 0; idx < nTrackCandidates; idx++) {
    short trackCandidateType = trackCandidates.trackCandidateType()[idx];
    std::vector<unsigned int> hit_idx = getHitIdxs(trackCandidateType, trackCandidates.hitIndices()[idx], hits.idxs());

    out_tc_hitIdxs_.push_back(hit_idx);
    out_tc_len_.push_back(hit_idx.size());
    out_tc_seedIdx_.push_back(trackCandidates.pixelSeedIndex()[idx]);
    out_tc_trackCandidateType_.push_back(trackCandidateType);
  }
}

void LST::run(Queue& queue,
              bool verbose,
              float const ptCut,
              LSTESData<Device> const* deviceESData,
              HitsHostCollection const* hitsHC,
              PixelSegmentsHostCollection const* pixelSegmentsHC,
              bool no_pls_dupclean,
              bool tc_pls_triplets) {
  auto event = LSTEvent(verbose, ptCut, queue, deviceESData);

  event.addHitToEvent(hitsHC);
  event.addPixelSegmentToEventStart(pixelSegmentsHC);
  event.createMiniDoublets();
  if (verbose) {
    alpaka::wait(queue);  // event calls are asynchronous: wait before printing
    printf("# of Mini-doublets produced: %d\n", event.getNumberOfMiniDoublets());
    printf("# of Mini-doublets produced barrel layer 1: %d\n", event.getNumberOfMiniDoubletsByLayerBarrel(0));
    printf("# of Mini-doublets produced barrel layer 2: %d\n", event.getNumberOfMiniDoubletsByLayerBarrel(1));
    printf("# of Mini-doublets produced barrel layer 3: %d\n", event.getNumberOfMiniDoubletsByLayerBarrel(2));
    printf("# of Mini-doublets produced barrel layer 4: %d\n", event.getNumberOfMiniDoubletsByLayerBarrel(3));
    printf("# of Mini-doublets produced barrel layer 5: %d\n", event.getNumberOfMiniDoubletsByLayerBarrel(4));
    printf("# of Mini-doublets produced barrel layer 6: %d\n", event.getNumberOfMiniDoubletsByLayerBarrel(5));
    printf("# of Mini-doublets produced endcap layer 1: %d\n", event.getNumberOfMiniDoubletsByLayerEndcap(0));
    printf("# of Mini-doublets produced endcap layer 2: %d\n", event.getNumberOfMiniDoubletsByLayerEndcap(1));
    printf("# of Mini-doublets produced endcap layer 3: %d\n", event.getNumberOfMiniDoubletsByLayerEndcap(2));
    printf("# of Mini-doublets produced endcap layer 4: %d\n", event.getNumberOfMiniDoubletsByLayerEndcap(3));
    printf("# of Mini-doublets produced endcap layer 5: %d\n", event.getNumberOfMiniDoubletsByLayerEndcap(4));
  }

  event.createSegmentsWithModuleMap();
  if (verbose) {
    alpaka::wait(queue);  // event calls are asynchronous: wait before printing
    printf("# of Segments produced: %d\n", event.getNumberOfSegments());
    printf("# of Segments produced layer 1-2:  %d\n", event.getNumberOfSegmentsByLayerBarrel(0));
    printf("# of Segments produced layer 2-3:  %d\n", event.getNumberOfSegmentsByLayerBarrel(1));
    printf("# of Segments produced layer 3-4:  %d\n", event.getNumberOfSegmentsByLayerBarrel(2));
    printf("# of Segments produced layer 4-5:  %d\n", event.getNumberOfSegmentsByLayerBarrel(3));
    printf("# of Segments produced layer 5-6:  %d\n", event.getNumberOfSegmentsByLayerBarrel(4));
    printf("# of Segments produced endcap layer 1:  %d\n", event.getNumberOfSegmentsByLayerEndcap(0));
    printf("# of Segments produced endcap layer 2:  %d\n", event.getNumberOfSegmentsByLayerEndcap(1));
    printf("# of Segments produced endcap layer 3:  %d\n", event.getNumberOfSegmentsByLayerEndcap(2));
    printf("# of Segments produced endcap layer 4:  %d\n", event.getNumberOfSegmentsByLayerEndcap(3));
    printf("# of Segments produced endcap layer 5:  %d\n", event.getNumberOfSegmentsByLayerEndcap(4));
  }

  event.createTriplets();
  if (verbose) {
    alpaka::wait(queue);  // event calls are asynchronous: wait before printing
    printf("# of T3s produced: %d\n", event.getNumberOfTriplets());
    printf("# of T3s produced layer 1-2-3: %d\n", event.getNumberOfTripletsByLayerBarrel(0));
    printf("# of T3s produced layer 2-3-4: %d\n", event.getNumberOfTripletsByLayerBarrel(1));
    printf("# of T3s produced layer 3-4-5: %d\n", event.getNumberOfTripletsByLayerBarrel(2));
    printf("# of T3s produced layer 4-5-6: %d\n", event.getNumberOfTripletsByLayerBarrel(3));
    printf("# of T3s produced endcap layer 1-2-3: %d\n", event.getNumberOfTripletsByLayerEndcap(0));
    printf("# of T3s produced endcap layer 2-3-4: %d\n", event.getNumberOfTripletsByLayerEndcap(1));
    printf("# of T3s produced endcap layer 3-4-5: %d\n", event.getNumberOfTripletsByLayerEndcap(2));
    printf("# of T3s produced endcap layer 1: %d\n", event.getNumberOfTripletsByLayerEndcap(0));
    printf("# of T3s produced endcap layer 2: %d\n", event.getNumberOfTripletsByLayerEndcap(1));
    printf("# of T3s produced endcap layer 3: %d\n", event.getNumberOfTripletsByLayerEndcap(2));
    printf("# of T3s produced endcap layer 4: %d\n", event.getNumberOfTripletsByLayerEndcap(3));
    printf("# of T3s produced endcap layer 5: %d\n", event.getNumberOfTripletsByLayerEndcap(4));
  }

  event.createQuintuplets();
  if (verbose) {
    alpaka::wait(queue);  // event calls are asynchronous: wait before printing
    printf("# of Quintuplets produced: %d\n", event.getNumberOfQuintuplets());
    printf("# of Quintuplets produced layer 1-2-3-4-5-6: %d\n", event.getNumberOfQuintupletsByLayerBarrel(0));
    printf("# of Quintuplets produced layer 2: %d\n", event.getNumberOfQuintupletsByLayerBarrel(1));
    printf("# of Quintuplets produced layer 3: %d\n", event.getNumberOfQuintupletsByLayerBarrel(2));
    printf("# of Quintuplets produced layer 4: %d\n", event.getNumberOfQuintupletsByLayerBarrel(3));
    printf("# of Quintuplets produced layer 5: %d\n", event.getNumberOfQuintupletsByLayerBarrel(4));
    printf("# of Quintuplets produced layer 6: %d\n", event.getNumberOfQuintupletsByLayerBarrel(5));
    printf("# of Quintuplets produced endcap layer 1: %d\n", event.getNumberOfQuintupletsByLayerEndcap(0));
    printf("# of Quintuplets produced endcap layer 2: %d\n", event.getNumberOfQuintupletsByLayerEndcap(1));
    printf("# of Quintuplets produced endcap layer 3: %d\n", event.getNumberOfQuintupletsByLayerEndcap(2));
    printf("# of Quintuplets produced endcap layer 4: %d\n", event.getNumberOfQuintupletsByLayerEndcap(3));
    printf("# of Quintuplets produced endcap layer 5: %d\n", event.getNumberOfQuintupletsByLayerEndcap(4));
  }

  // event.addPixelSegmentToEventFinalize(
  //     in_hitIndices_vec0_, in_hitIndices_vec1_, in_hitIndices_vec2_, in_hitIndices_vec3_, in_deltaPhi_vec_);

  event.pixelLineSegmentCleaning(no_pls_dupclean);

  event.createPixelQuintuplets();
  if (verbose) {
    alpaka::wait(queue);  // event calls are asynchronous: wait before printing
    printf("# of Pixel Quintuplets produced: %d\n", event.getNumberOfPixelQuintuplets());
  }

  event.createPixelTriplets();
  if (verbose) {
    alpaka::wait(queue);  // event calls are asynchronous: wait before printing
    printf("# of Pixel T3s produced: %d\n", event.getNumberOfPixelTriplets());
  }

  event.createTrackCandidates(no_pls_dupclean, tc_pls_triplets);
  if (verbose) {
    alpaka::wait(queue);  // event calls are asynchronous: wait before printing
    printf("# of TrackCandidates produced: %d\n", event.getNumberOfTrackCandidates());
    printf("        # of Pixel TrackCandidates produced: %d\n", event.getNumberOfPixelTrackCandidates());
    printf("        # of pT5 TrackCandidates produced: %d\n", event.getNumberOfPT5TrackCandidates());
    printf("        # of pT3 TrackCandidates produced: %d\n", event.getNumberOfPT3TrackCandidates());
    printf("        # of pLS TrackCandidates produced: %d\n", event.getNumberOfPLSTrackCandidates());
    printf("        # of T5 TrackCandidates produced: %d\n", event.getNumberOfT5TrackCandidates());
  }

  getOutput(event);
}
