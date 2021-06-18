#ifndef CUDADataFormats_Track_TrackHeterogeneousT_H
#define CUDADataFormats_Track_TrackHeterogeneousT_H

#include <string>
#include <algorithm>

#include "CUDADataFormats/Track/interface/TrajectoryStateSoAT.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"

#include "CUDADataFormats/Common/interface/HeterogeneousSoA.h"

namespace pixelTrack {
  enum class Quality : uint8_t { bad = 0, edup, dup, loose, strict, tight, highPurity, notQuality };
  constexpr uint32_t qualitySize{uint8_t(Quality::notQuality)};
  const std::string qualityName[qualitySize]{"bad", "edup", "dup", "loose", "strict", "tight", "highPurity"};
  inline Quality qualityByName(std::string const &name) {
    auto qp = std::find(qualityName, qualityName + qualitySize, name) - qualityName;
    return static_cast<Quality>(qp);
  }
}  // namespace pixelTrack

template <int32_t S>
class TrackSoAHeterogeneousT {
public:
  static constexpr int32_t stride() { return S; }

  using Quality = pixelTrack::Quality;
  using hindex_type = uint32_t;
  using HitContainer = cms::cuda::OneToManyAssoc<hindex_type, S + 1, 5 * S>;

  // Always check quality is at least loose!
  // CUDA does not support enums  in __lgc ...
private:
  eigenSoA::ScalarSoA<uint8_t, S> quality_;

public:
  constexpr Quality quality(int32_t i) const { return (Quality)(quality_(i)); }
  constexpr Quality &quality(int32_t i) { return (Quality &)(quality_(i)); }
  constexpr Quality const *qualityData() const { return (Quality const *)(quality_.data()); }
  constexpr Quality *qualityData() { return (Quality *)(quality_.data()); }

  // this is chi2/ndof as not necessarely all hits are used in the fit
  eigenSoA::ScalarSoA<float, S> chi2;

  constexpr int nHits(int i) const { return detIndices.size(i); }

  // State at the Beam spot
  // phi,tip,1/pt,cotan(theta),zip
  TrajectoryStateSoAT<S> stateAtBS;
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
};

namespace pixelTrack {

#ifdef GPU_SMALL_EVENTS
  // kept for testing and debugging
  constexpr uint32_t maxNumber() { return 2 * 1024; }
#else
  // tested on MC events with 55-75 pileup events
  constexpr uint32_t maxNumber() { return 32 * 1024; }
#endif

  using TrackSoA = TrackSoAHeterogeneousT<maxNumber()>;
  using TrajectoryState = TrajectoryStateSoAT<maxNumber()>;
  using HitContainer = TrackSoA::HitContainer;

}  // namespace pixelTrack

#endif  // CUDADataFormats_Track_TrackHeterogeneousT_H
