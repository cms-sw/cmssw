#include "RecoTracker/LSTCore/interface/alpaka/LST.h"

#include "LSTEvent.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE::lst;

#include "Math/Vector3D.h"
#include "Math/VectorUtil.h"
using XYZVector = ROOT::Math::XYZVector;

void LST::run(Queue& queue,
              bool verbose,
              float const ptCut,
              uint16_t const clustSizeCut,
              LSTESData<Device> const* deviceESData,
              LSTInputDeviceCollection const* lstInputDC,
              bool no_pls_dupclean,
              bool tc_pls_triplets) {
  auto event = LSTEvent(verbose, ptCut, clustSizeCut, queue, deviceESData);

  event.addInputToEvent(lstInputDC);
  event.addHitToEvent();
  event.addPixelSegmentToEventStart();
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

  event.addPixelSegmentToEventFinalize();

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

  event.createQuadruplets();
  if (verbose) {
    alpaka::wait(queue);  // event calls are asynchronous: wait before printing
    printf("# of Quadruplets produced: %d\n", event.getNumberOfQuadruplets());
    printf("# of Quadruplets produced layer 1-2-3-4: %d\n", event.getNumberOfQuadrupletsByLayerBarrel(0));
    printf("# of Quadruplets produced layer 2: %d\n", event.getNumberOfQuadrupletsByLayerBarrel(1));
    printf("# of Quadruplets produced layer 3: %d\n", event.getNumberOfQuadrupletsByLayerBarrel(2));
    printf("# of Quadruplets produced layer 4: %d\n", event.getNumberOfQuadrupletsByLayerBarrel(3));
    printf("# of Quadruplets produced layer 5: %d\n", event.getNumberOfQuadrupletsByLayerBarrel(4));
    printf("# of Quadruplets produced layer 6: %d\n", event.getNumberOfQuadrupletsByLayerBarrel(5));
    printf("# of Quadruplets produced endcap layer 1: %d\n", event.getNumberOfQuadrupletsByLayerEndcap(0));
    printf("# of Quadruplets produced endcap layer 2: %d\n", event.getNumberOfQuadrupletsByLayerEndcap(1));
    printf("# of Quadruplets produced endcap layer 3: %d\n", event.getNumberOfQuadrupletsByLayerEndcap(2));
    printf("# of Quadruplets produced endcap layer 4: %d\n", event.getNumberOfQuadrupletsByLayerEndcap(3));
    printf("# of Quadruplets produced endcap layer 5: %d\n", event.getNumberOfQuadrupletsByLayerEndcap(4));
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
    printf("        # of T4 TrackCandidates produced: %d\n", event.getNumberOfT4TrackCandidates());
  }

  trackCandidatesBaseDC_ = event.releaseTrackCandidatesBaseDeviceCollection();
}
