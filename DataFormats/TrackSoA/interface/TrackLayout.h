#ifndef DataFormats_Track_interface_TrackLayout_h
#define DataFormats_Track_interface_TrackLayout_h

#include <Eigen/Core>
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"

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

  using HitContainer = cms::alpakatools::OneToManyAssoc<hindex_type, S + 1, H * S>;

  GENERATE_SOA_LAYOUT(TrackSoAHeterogeneousLayout,
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
using TrackLayout = typename TrackSoA<TrackerTraits>::template TrackSoAHeterogeneousLayout<>;
template <typename TrackerTraits>
using TrackSoAView = typename TrackSoA<TrackerTraits>::template TrackSoAHeterogeneousLayout<>::View;
template <typename TrackerTraits>
using TrackSoAConstView = typename TrackSoA<TrackerTraits>::template TrackSoAHeterogeneousLayout<>::ConstView;

  // State at the Beam spot
  // phi,tip,1/pt,cotan(theta),zip
//  template <typename TrackerTraits>
//  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr float charge(const TrackSoAConstView<TrackerTraits>& tracks, int32_t i) {
//    float v = tracks[i].state()(2);
//    return float((0.0f < v) - (v < 0.0f));
//  }   
  
#endif
