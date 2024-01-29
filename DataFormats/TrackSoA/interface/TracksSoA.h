#ifndef DataFormats_Track_interface_TrackLayout_h
#define DataFormats_Track_interface_TrackLayout_h

#include <Eigen/Core>
#include "HeterogeneousCore/AlpakaInterface/interface/OneToManyAssoc.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"

namespace reco {

  template <typename TrackerTraits>
  struct TrackSoA {
    static constexpr int32_t S = TrackerTraits::maxNumberOfTuples;
    static constexpr int32_t H = TrackerTraits::avgHitsPerTrack;
    // Aliases in order to not confuse the GENERATE_SOA_LAYOUT
    // macro with weird colons and angled brackets.
    using Vector5f = Eigen::Matrix<float, 5, 1>;
    using Vector15f = Eigen::Matrix<float, 15, 1>;
    using Quality = pixelTrack::Quality;

    using hindex_type = uint32_t;

    using HitContainer = cms::alpakatools::OneToManyAssocSequential<hindex_type, S + 1, H * S>;

    GENERATE_SOA_LAYOUT(Layout,
                        SOA_COLUMN(Quality, quality),
                        SOA_COLUMN(float, chi2),
                        SOA_COLUMN(int8_t, nLayers),
                        SOA_COLUMN(float, eta),
                        SOA_COLUMN(float, pt),
                        SOA_EIGEN_COLUMN(Vector5f, state),
                        SOA_EIGEN_COLUMN(Vector15f, covariance),
                        SOA_SCALAR(int, nTracks),
                        SOA_SCALAR(HitContainer, hitIndices),
                        SOA_SCALAR(HitContainer, detIndices))
  };

  template <typename TrackerTraits>
  using TrackLayout = typename reco::TrackSoA<TrackerTraits>::template Layout<>;
  template <typename TrackerTraits>
  using TrackSoAView = typename reco::TrackSoA<TrackerTraits>::template Layout<>::View;
  template <typename TrackerTraits>
  using TrackSoAConstView = typename reco::TrackSoA<TrackerTraits>::template Layout<>::ConstView;

  template <typename TrackerTraits>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr float charge(const TrackSoAConstView<TrackerTraits> &tracks,
                                                                    int32_t i) {
    //was: std::copysign(1.f, tracks[i].state()(2)). Will be constexpr with C++23
    float v = tracks[i].state()(2);
    return float((0.0f < v) - (v < 0.0f));
  }

}  // namespace reco

#endif
