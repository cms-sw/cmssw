#ifndef CUDADataFormats_Track_TrackHeterogeneousT_H
#define CUDADataFormats_Track_TrackHeterogeneousT_H

#include <string>
#include <algorithm>

#include "CUDADataFormats/Track/interface/TrajectoryStateSoAT.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "CUDADataFormats/Common/interface/HeterogeneousSoA.h"
#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"

namespace pixelTrack {

  enum class Quality : uint8_t { bad = 0, edup, dup, loose, strict, tight, highPurity, notQuality };
  constexpr uint32_t qualitySize{uint8_t(Quality::notQuality)};
  const std::string qualityName[qualitySize]{"bad", "edup", "dup", "loose", "strict", "tight", "highPurity"};
  inline Quality qualityByName(std::string const &name) {
    auto qp = std::find(qualityName, qualityName + qualitySize, name) - qualityName;
    return static_cast<Quality>(qp);
  }

}  // namespace pixelTrack

template <typename TrackerTraits>
class TrackSoAHeterogeneousT {
public:
  static constexpr int32_t S = TrackerTraits::maxNumberOfTuples;
  static constexpr int32_t H = TrackerTraits::maxHitsOnTrack;  // Average hits rather than max?
  static constexpr int32_t stride() { return S; }

  using hindex_type = uint32_t;  //TrackerTraits::hindex_type ?

  using Quality = pixelTrack::Quality;
  using HitContainer = cms::cuda::OneToManyAssoc<hindex_type, S + 1, H * S>;

  // Always check quality is at least loose!
  // CUDA does not support enums  in __lgc ...
protected:
  eigenSoA::ScalarSoA<uint8_t, S> quality_;

public:
  constexpr Quality quality(int32_t i) const { return (Quality)(quality_(i)); }
  constexpr Quality &quality(int32_t i) { return (Quality &)(quality_(i)); }
  constexpr Quality const *qualityData() const { return (Quality const *)(quality_.data()); }
  constexpr Quality *qualityData() { return (Quality *)(quality_.data()); }

  // this is chi2/ndof as not necessarely all hits are used in the fit
  eigenSoA::ScalarSoA<float, S> chi2;

  eigenSoA::ScalarSoA<int8_t, S> nLayers;

  constexpr int nTracks() const { return nTracks_; }
  constexpr void setNTracks(int n) { nTracks_ = n; }

  constexpr int nHits(int i) const { return detIndices.size(i); }

  constexpr bool isTriplet(int i) const { return nLayers(i) == 3; }

  constexpr int computeNumberOfLayers(int32_t i) const {
    // layers are in order and we assume tracks are either forward or backward
    auto pdet = detIndices.begin(i);
    int nl = 1;
    auto ol = pixelTopology::getLayer<TrackerTraits>(*pdet);
    for (; pdet < detIndices.end(i); ++pdet) {
      auto il = pixelTopology::getLayer<TrackerTraits>(*pdet);
      if (il != ol)
        ++nl;
      ol = il;
    }
    return nl;
  }

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

private:
  int nTracks_;
};

namespace pixelTrack {

  template <typename TrackerTraits>
  using TrackSoAT = TrackSoAHeterogeneousT<TrackerTraits>;

  template <typename TrackerTraits>
  using HitContainerT = typename TrackSoAHeterogeneousT<TrackerTraits>::HitContainer;

  //Used only to ease classes definitions
  using TrackSoAPhase1 = TrackSoAHeterogeneousT<pixelTopology::Phase1>;
  using TrackSoAPhase2 = TrackSoAHeterogeneousT<pixelTopology::Phase2>;

  template <typename TrackerTraits, typename Enable = void>
  struct QualityCutsT {};

  template <typename TrackerTraits>
  struct QualityCutsT<TrackerTraits, pixelTopology::isPhase1Topology<TrackerTraits>> {
    // chi2 cut = chi2Scale * (chi2Coeff[0] + pT/GeV * (chi2Coeff[1] + pT/GeV * (chi2Coeff[2] + pT/GeV * chi2Coeff[3])))
    float chi2Coeff[4];
    float chi2MaxPt;  // GeV
    float chi2Scale;

    struct Region {
      float maxTip;  // cm
      float minPt;   // GeV
      float maxZip;  // cm
    };

    Region triplet;
    Region quadruplet;

    __device__ __forceinline__ bool isHP(TrackSoAHeterogeneousT<TrackerTraits> const *__restrict__ tracks,
                                         int nHits,
                                         int it) const {
      // impose "region cuts" based on the fit results (phi, Tip, pt, cotan(theta)), Zip)
      // default cuts:
      //   - for triplets:    |Tip| < 0.3 cm, pT > 0.5 GeV, |Zip| < 12.0 cm
      //   - for quadruplets: |Tip| < 0.5 cm, pT > 0.3 GeV, |Zip| < 12.0 cm
      // (see CAHitNtupletGeneratorGPU.cc)
      auto const &region = (nHits > 3) ? quadruplet : triplet;
      return (std::abs(tracks->tip(it)) < region.maxTip) and (tracks->pt(it) > region.minPt) and
             (std::abs(tracks->zip(it)) < region.maxZip);
    }

    __device__ __forceinline__ bool strictCut(TrackSoAHeterogeneousT<TrackerTraits> const *__restrict__ tracks,
                                              int it) const {
      auto roughLog = [](float x) {
        // max diff [0.5,12] at 1.25 0.16143
        // average diff  0.0662998
        union IF {
          uint32_t i;
          float f;
        };
        IF z;
        z.f = x;
        uint32_t lsb = 1 < 21;
        z.i += lsb;
        z.i >>= 21;
        auto f = z.i & 3;
        int ex = int(z.i >> 2) - 127;

        // log2(1+0.25*f)
        // averaged over bins
        const float frac[4] = {0.160497f, 0.452172f, 0.694562f, 0.901964f};
        return float(ex) + frac[f];
      };

      float pt = std::min<float>(tracks->pt(it), chi2MaxPt);
      float chi2Cut = chi2Scale * (chi2Coeff[0] + roughLog(pt) * chi2Coeff[1]);
      if (tracks->chi2(it) >= chi2Cut) {
#ifdef NTUPLE_FIT_DEBUG
        printf("Bad chi2 %d pt %f eta %f chi2 %f\n", it, tracks->pt(it), tracks->eta(it), tracks->chi2(it));
#endif
        return true;
      }
      return false;
    }
  };

  template <typename TrackerTraits>
  struct QualityCutsT<TrackerTraits, pixelTopology::isPhase2Topology<TrackerTraits>> {
    float maxChi2;
    float minPt;
    float maxTip;
    float maxZip;

    __device__ __forceinline__ bool isHP(TrackSoAHeterogeneousT<TrackerTraits> const *__restrict__ tracks,
                                         int nHits,
                                         int it) const {
      return (std::abs(tracks->tip(it)) < maxTip) and (tracks->pt(it) > minPt) and (std::abs(tracks->zip(it)) < maxZip);
    }
    __device__ __forceinline__ bool strictCut(TrackSoAHeterogeneousT<TrackerTraits> const *__restrict__ tracks,
                                              int it) const {
      return tracks->chi2(it) >= maxChi2;
    }
  };

}  // namespace pixelTrack

#endif  // CUDADataFormats_Track_TrackHeterogeneousT_H
