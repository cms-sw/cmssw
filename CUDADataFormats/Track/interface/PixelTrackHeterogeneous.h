#ifndef CUDADataFormatsTrackTrackHeterogeneous_H
#define CUDADataFormatsTrackTrackHeterogeneous_H

#include "CUDADataFormats/Track/interface/TrajectoryStateSoA.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"

#include "CUDADataFormats/Common/interface/HeterogeneousSoA.h"

namespace trackQuality {
  enum Quality : uint8_t { bad = 0, dup, loose, strict, tight, highPurity };
}

template <int32_t S>
class TrackSoAT {
public:
  static constexpr int32_t stride() { return S; }

  using Quality = trackQuality::Quality;
  using hindex_type = uint32_t;
  using HitContainer = cms::cuda::OneToManyAssoc<hindex_type, S, 5 * S>;

  // Always check quality is at least loose!
  // CUDA does not support enums  in __lgc ...
  eigenSoA::ScalarSoA<uint8_t, S> m_quality;
  constexpr Quality quality(int32_t i) const { return (Quality)(m_quality(i)); }
  constexpr Quality &quality(int32_t i) { return (Quality &)(m_quality(i)); }
  constexpr Quality const *qualityData() const { return (Quality const *)(m_quality.data()); }
  constexpr Quality *qualityData() { return (Quality *)(m_quality.data()); }

  // this is chi2/ndof as not necessarely all hits are used in the fit
  eigenSoA::ScalarSoA<float, S> chi2;

  constexpr int nHits(int i) const { return detIndices.size(i); }

  // State at the Beam spot
  // phi,tip,1/pt,cotan(theta),zip
  TrajectoryStateSoA<S> stateAtBS;
  eigenSoA::ScalarSoA<float, S> eta;
  eigenSoA::ScalarSoA<float, S> pt;
  constexpr float charge(int32_t i) const { return std::copysign(1.f, stateAtBS.state(i)(2)); }
  constexpr float phi(int32_t i) const { return stateAtBS.state(i)(0); }
  constexpr float tip(int32_t i) const { return stateAtBS.state(i)(1); }
  constexpr float zip(int32_t i) const { return stateAtBS.state(i)(4); }

  // state at the detector of the outermost hit
  // representation to be decided...
  // not yet filled on GPU
  // TrajectoryStateSoA<S> stateAtOuterDet;

  HitContainer hitIndices;
  HitContainer detIndices;

  // total number of tracks (including those not fitted)
  uint32_t m_nTracks;
};

namespace pixelTrack {

#ifdef GPU_SMALL_EVENTS
  // kept for testing and debugging
  constexpr uint32_t maxNumber() { return 2 * 1024; }
#else
  // tested on MC events with 55-75 pileup events
  constexpr uint32_t maxNumber() { return 32 * 1024; }
#endif

  using TrackSoA = TrackSoAT<maxNumber()>;
  using TrajectoryState = TrajectoryStateSoA<maxNumber()>;
  using HitContainer = TrackSoA::HitContainer;
  using Quality = trackQuality::Quality;

}  // namespace pixelTrack

using PixelTrackHeterogeneous = HeterogeneousSoA<pixelTrack::TrackSoA>;

#endif  // CUDADataFormatsTrackTrackSoA_H
