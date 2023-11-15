#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"

using namespace reco;

using Quality = pixelTrack::Quality;
namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace cms::alpakatools;
  namespace testTrackSoA {

    // Kernel which fills the TrackSoAView with data
    // to test writing to it
    template <typename TrackerTraits>
    class TestFillKernel {
    public:
      template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
      ALPAKA_FN_ACC void operator()(TAcc const& acc, TrackSoAView<TrackerTraits> tracks_view) const {
        if (cms::alpakatools::once_per_grid(acc)) {
          tracks_view.nTracks() = 420;
        }

        for (int32_t j : elements_with_stride(acc, tracks_view.metadata().size())) {
          tracks_view[j].pt() = (float)j;
          tracks_view[j].eta() = (float)j;
          tracks_view[j].chi2() = (float)j;
          tracks_view[j].quality() = (Quality)(j % 256);
          tracks_view[j].nLayers() = j % 128;
          tracks_view.hitIndices().off[j] = j;
        }
      }
    };

    // Kernel which reads from the TrackSoAView to verify
    // that it was written correctly from the fill kernel
    template <typename TrackerTraits>
    class TestVerifyKernel {
    public:
      template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
      ALPAKA_FN_ACC void operator()(TAcc const& acc, TrackSoAConstView<TrackerTraits> tracks_view) const {
        if (cms::alpakatools::once_per_grid(acc)) {
          ALPAKA_ASSERT_OFFLOAD(tracks_view.nTracks() == 420);
        }
        for (int32_t j : elements_with_stride(acc, tracks_view.nTracks())) {
          assert(abs(tracks_view[j].pt() - (float)j) < .0001);
          assert(abs(tracks_view[j].eta() - (float)j) < .0001);
          assert(abs(tracks_view[j].chi2() - (float)j) < .0001);
          assert(tracks_view[j].quality() == (Quality)(j % 256));
          assert(tracks_view[j].nLayers() == j % 128);
          assert(tracks_view.hitIndices().off[j] == uint32_t(j));
        }
      }
    };

    // Host function which invokes the two kernels above
    template <typename TrackerTraits>
    void runKernels(TrackSoAView<TrackerTraits> tracks_view, Queue& queue) {
      uint32_t items = 64;
      uint32_t groups = divide_up_by(tracks_view.metadata().size(), items);
      auto workDiv = make_workdiv<Acc1D>(groups, items);
      alpaka::exec<Acc1D>(queue, workDiv, TestFillKernel<TrackerTraits>{}, tracks_view);
      alpaka::exec<Acc1D>(queue,
                          workDiv,
                          TestVerifyKernel<TrackerTraits>{},
                          tracks_view);  //TODO: wait for some PR that solves this and then check it!!!
    }

    template void runKernels<pixelTopology::Phase1>(TrackSoAView<pixelTopology::Phase1> tracks_view, Queue& queue);
    template void runKernels<pixelTopology::Phase2>(TrackSoAView<pixelTopology::Phase2> tracks_view, Queue& queue);

  }  // namespace testTrackSoA
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
