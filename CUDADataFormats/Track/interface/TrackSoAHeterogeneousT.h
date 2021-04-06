#ifndef CUDADataFormats_Track_PixelTrackHeterogeneousT_h
#define CUDADataFormats_Track_PixelTrackHeterogeneousT_h

#include "CUDADataFormats/Track/interface/TrajectoryStateSoAT.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"

#include "CUDADataFormats/Common/interface/HeterogeneousSoA.h"

namespace pixelTrack {
  enum class Quality : uint8_t { bad = 0, dup, loose, strict, tight, highPurity };
}

template <int32_t S>
class TrackSoAHeterogeneousT {
public:
  static constexpr int32_t stride() { return S; }

  using Quality = pixelTrack::Quality;
  using hindex_type = uint32_t;
  using HitContainer = cms::cuda::OneToManyAssoc<hindex_type, S + 1, 5 * S>;


  // quality accessors
  constexpr Quality quality(int32_t i) const { return reinterpret_cast<Quality>(quality_(i)); }
  constexpr Quality &quality(int32_t i) { return reinterpret_cast<Quality &>(quality_(i)); }
  constexpr Quality const *qualityData() const { return reinterpret_cast<Quality const *>(quality_.data()); }
  constexpr Quality *qualityData() { return reinterpret_cast<Quality *>(quality_.data()); }

  // chi2 accessors
  constexpr auto & chi2(int32_t i) { return chi2_(i); }
  constexpr auto chi2(int32_t i) const { return chi2_(i); }

  // stateAtBS accessors
  constexpr auto & stateAtBS() { return stateAtBS_; }
  constexpr auto stateAtBS() const { return stateAtBS_; }
  // eta accessors
  constexpr auto & eta(int32_t i) { return eta_(i); }
  constexpr auto eta(int32_t i) const { return eta_(i); }
  // pt accessors
  constexpr auto & pt(int32_t i) { return pt_(i); }
  constexpr auto pt(int32_t i) const { return pt_(i); }

  constexpr float charge(int32_t i) const { return std::copysign(1.f, stateAtBS_.state(i)(2)); }
  constexpr float phi(int32_t i) const { return stateAtBS_.state(i)(0); }
  constexpr float tip(int32_t i) const { return stateAtBS_.state(i)(1); }
  constexpr float zip(int32_t i) const { return stateAtBS_.state(i)(4); }

  // hitIndices accessors
  constexpr auto & hitIndices() { return hitIndices_; }
  constexpr auto const & hitIndices() const { return hitIndices_; }

  // detInndices accessor
  constexpr int nHits(int i) const { return detIndices_.size(i); }
  constexpr auto & detIndices() { return detIndices_; }
  constexpr auto const & detIndices() const { return detIndices_; }

  // state at the detector of the outermost hit
  // representation to be decided...
  // not yet filled on GPU
  // TrajectoryStateSoA<S> stateAtOuterDet;
private:
  // Always check quality is at least loose!
  // CUDA does not support enums  in __lgc ...
  eigenSoA::ScalarSoA<uint8_t, S> quality_;

  // this is chi2/ndof as not necessarely all hits are used in the fit
  eigenSoA::ScalarSoA<float, S> chi2_;

  // State at the Beam spot
  // phi,tip,1/pt,cotan(theta),zip
  TrajectoryStateSoAT<S> stateAtBS_;
  eigenSoA::ScalarSoA<float, S> eta_;
  eigenSoA::ScalarSoA<float, S> pt_;

  HitContainer hitIndices_;
  HitContainer detIndices_;
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

#endif  // CUDADataFormats_Track_PixelTrackHeterogeneousT_h
