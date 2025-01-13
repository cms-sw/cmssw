#ifndef DataFormats_TrackSoA_interface_TracksSoA_h
#define DataFormats_TrackSoA_interface_TracksSoA_h

#include <alpaka/alpaka.hpp>

#include <Eigen/Core>

#include "HeterogeneousCore/AlpakaInterface/interface/OneToManyAssoc.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "Geometry/CommonTopologies/interface/SimplePixelStripTopology.h"
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
                        // state at the beam spot: {phi, tip, 1/pt, cotan(theta), zip}
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

  /* Implement a type trait to identify the specialisations of TrackSoAConstView<TrackerTraits>
   *
   * This is done explicitly for all possible pixel topologies, because we did not find a way
   * to use template deduction with a partial specialisation.
   */
  template <typename T>
  struct IsTrackSoAConstView : std::false_type {};
  template <>
  struct IsTrackSoAConstView<TrackSoAConstView<pixelTopology::Phase1>> : std::true_type {};
  template <>
  struct IsTrackSoAConstView<TrackSoAView<pixelTopology::Phase1>> : std::true_type {};
  template <>
  struct IsTrackSoAConstView<TrackSoAConstView<pixelTopology::Phase2>> : std::true_type {};
  template <>
  struct IsTrackSoAConstView<TrackSoAView<pixelTopology::Phase2>> : std::true_type {};
  template <>
  struct IsTrackSoAConstView<TrackSoAConstView<pixelTopology::HIonPhase1>> : std::true_type {};
  template <>
  struct IsTrackSoAConstView<TrackSoAView<pixelTopology::HIonPhase1>> : std::true_type {};
  template <>
  struct IsTrackSoAConstView<TrackSoAConstView<pixelTopology::Phase1Strip>> : std::true_type {};
  template <>
  struct IsTrackSoAConstView<TrackSoAView<pixelTopology::Phase1Strip>> : std::true_type {};

  template <typename T>
  constexpr bool isTrackSoAConstView = IsTrackSoAConstView<T>::value;

  // enable_if should be used when there is another implementation,
  // please use static_assert to report invalid template arguments
  template <typename ConstView>  //, typename = std::enable_if_t<isTrackSoAConstView<ConstView>>>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr float charge(ConstView const& tracks, int32_t i) {
    //was: std::copysign(1.f, tracks[i].state()(2)). Will be constexpr with C++23
    float v = tracks[i].state()(2);
    return float((0.0f < v) - (v < 0.0f));
  }

  template <typename ConstView>  //, typename = std::enable_if_t<isTrackSoAConstView<ConstView>>>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr float phi(ConstView const& tracks, int32_t i) {
    return tracks[i].state()(0);
  }

  template <typename ConstView>  //, typename = std::enable_if_t<isTrackSoAConstView<ConstView>>>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr float tip(ConstView const& tracks, int32_t i) {
    return tracks[i].state()(1);
  }

  template <typename ConstView>  //, typename = std::enable_if_t<isTrackSoAConstView<ConstView>>>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr float zip(ConstView const& tracks, int32_t i) {
    return tracks[i].state()(4);
  }

  template <typename ConstView>  //, typename = std::enable_if_t<isTrackSoAConstView<ConstView>>>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr bool isTriplet(ConstView const& tracks, int32_t i) {
    return tracks[i].nLayers() == 3;
  }

}  // namespace reco

#endif  // DataFormats_TrackSoA_interface_TracksSoA_h
