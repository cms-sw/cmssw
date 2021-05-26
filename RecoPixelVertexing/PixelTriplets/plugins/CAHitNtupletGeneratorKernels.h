#ifndef RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorKernels_h
#define RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorKernels_h

// #define GPU_DEBUG

#include "CUDADataFormats/Track/interface/PixelTrackHeterogeneous.h"
#include "GPUCACell.h"

// #define DUMP_GPU_TK_TUPLES

namespace cAHitNtupletGenerator {

  // counters
  struct Counters {
    unsigned long long nEvents;
    unsigned long long nHits;
    unsigned long long nCells;
    unsigned long long nTuples;
    unsigned long long nFitTracks;
    unsigned long long nLooseTracks;
    unsigned long long nGoodTracks;
    unsigned long long nUsedHits;
    unsigned long long nDupHits;
    unsigned long long nKilledCells;
    unsigned long long nEmptyCells;
    unsigned long long nZeroTrackCells;
  };

  using HitsView = TrackingRecHit2DSOAView;
  using HitsOnGPU = TrackingRecHit2DSOAView;

  using HitToTuple = caConstants::HitToTuple;
  using TupleMultiplicity = caConstants::TupleMultiplicity;

  using Quality = pixelTrack::Quality;
  using TkSoA = pixelTrack::TrackSoA;
  using HitContainer = pixelTrack::HitContainer;

  struct QualityCuts {
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
  };

  // params (FIXME: thi si a POD: so no constructor no traling _  and no const as params_ is already const)
  struct Params {
    Params(bool onGPU,
           uint32_t minHitsPerNtuplet,
           uint32_t maxNumberOfDoublets,
           uint16_t minHitsForSharingCuts,
           bool useRiemannFit,
           bool fit5as4,
           bool includeJumpingForwardDoublets,
           bool earlyFishbone,
           bool lateFishbone,
           bool idealConditions,
           bool doStats,
           bool doClusterCut,
           bool doZ0Cut,
           bool doPtCut,
           bool doSharedHitCut,
           bool dupPassThrough,
           bool useSimpleTripletCleaner,
           float ptmin,
           float CAThetaCutBarrel,
           float CAThetaCutForward,
           float hardCurvCut,
           float dcaCutInnerTriplet,
           float dcaCutOuterTriplet,

           QualityCuts const& cuts)
        : onGPU_(onGPU),
          minHitsPerNtuplet_(minHitsPerNtuplet),
          maxNumberOfDoublets_(maxNumberOfDoublets),
          minHitsForSharingCut_(minHitsForSharingCuts),
          useRiemannFit_(useRiemannFit),
          fit5as4_(fit5as4),
          includeJumpingForwardDoublets_(includeJumpingForwardDoublets),
          earlyFishbone_(earlyFishbone),
          lateFishbone_(lateFishbone),
          idealConditions_(idealConditions),
          doStats_(doStats),
          doClusterCut_(doClusterCut),
          doZ0Cut_(doZ0Cut),
          doPtCut_(doPtCut),
          doSharedHitCut_(doSharedHitCut),
          dupPassThrough_(dupPassThrough),
          useSimpleTripletCleaner_(useSimpleTripletCleaner),
          ptmin_(ptmin),
          CAThetaCutBarrel_(CAThetaCutBarrel),
          CAThetaCutForward_(CAThetaCutForward),
          hardCurvCut_(hardCurvCut),
          dcaCutInnerTriplet_(dcaCutInnerTriplet),
          dcaCutOuterTriplet_(dcaCutOuterTriplet),
          cuts_(cuts) {}

    const bool onGPU_;
    const uint32_t minHitsPerNtuplet_;
    const uint32_t maxNumberOfDoublets_;
    const uint16_t minHitsForSharingCut_;
    const bool useRiemannFit_;
    const bool fit5as4_;
    const bool includeJumpingForwardDoublets_;
    const bool earlyFishbone_;
    const bool lateFishbone_;
    const bool idealConditions_;
    const bool doStats_;
    const bool doClusterCut_;
    const bool doZ0Cut_;
    const bool doPtCut_;
    const bool doSharedHitCut_;
    const bool dupPassThrough_;
    const bool useSimpleTripletCleaner_;
    const float ptmin_;
    const float CAThetaCutBarrel_;
    const float CAThetaCutForward_;
    const float hardCurvCut_;
    const float dcaCutInnerTriplet_;
    const float dcaCutOuterTriplet_;

    // quality cuts
    QualityCuts cuts_{// polynomial coefficients for the pT-dependent chi2 cut
                      {0.68177776, 0.74609577, -0.08035491, 0.00315399},
                      // max pT used to determine the chi2 cut
                      10.,
                      // chi2 scale factor: 30 for broken line fit, 45 for Riemann fit
                      30.,
                      // regional cuts for triplets
                      {
                          0.3,  // |Tip| < 0.3 cm
                          0.5,  // pT > 0.5 GeV
                          12.0  // |Zip| < 12.0 cm
                      },
                      // regional cuts for quadruplets
                      {
                          0.5,  // |Tip| < 0.5 cm
                          0.3,  // pT > 0.3 GeV
                          12.0  // |Zip| < 12.0 cm
                      }};

  };  // Params

}  // namespace cAHitNtupletGenerator

template <typename TTraits>
class CAHitNtupletGeneratorKernels {
public:
  using Traits = TTraits;

  using QualityCuts = cAHitNtupletGenerator::QualityCuts;
  using Params = cAHitNtupletGenerator::Params;
  using Counters = cAHitNtupletGenerator::Counters;

  template <typename T>
  using unique_ptr = typename Traits::template unique_ptr<T>;

  using HitsView = TrackingRecHit2DSOAView;
  using HitsOnGPU = TrackingRecHit2DSOAView;
  using HitsOnCPU = TrackingRecHit2DHeterogeneous<Traits>;

  using HitToTuple = caConstants::HitToTuple;
  using TupleMultiplicity = caConstants::TupleMultiplicity;

  using Quality = pixelTrack::Quality;
  using TkSoA = pixelTrack::TrackSoA;
  using HitContainer = pixelTrack::HitContainer;

  CAHitNtupletGeneratorKernels(Params const& params)
      : params_(params), paramsMaxDoubletes3Quarters_(3 * params.maxNumberOfDoublets_ / 4) {}
  ~CAHitNtupletGeneratorKernels() = default;

  TupleMultiplicity const* tupleMultiplicity() const { return device_tupleMultiplicity_.get(); }

  void launchKernels(HitsOnCPU const& hh, TkSoA* tuples_d, cudaStream_t cudaStream);

  void classifyTuples(HitsOnCPU const& hh, TkSoA* tuples_d, cudaStream_t cudaStream);

  void fillHitDetIndices(HitsView const* hv, TkSoA* tuples_d, cudaStream_t cudaStream);

  void buildDoublets(HitsOnCPU const& hh, cudaStream_t stream);
  void allocateOnGPU(int32_t nHits, cudaStream_t stream);
  void cleanup(cudaStream_t cudaStream);

  static void printCounters(Counters const* counters);
  void setCounters(Counters* counters) { counters_ = counters; }

private:
  Counters* counters_ = nullptr;

  // workspace
  unique_ptr<unsigned char[]> cellStorage_;
  unique_ptr<caConstants::CellNeighborsVector> device_theCellNeighbors_;
  caConstants::CellNeighbors* device_theCellNeighborsContainer_;
  unique_ptr<caConstants::CellTracksVector> device_theCellTracks_;
  caConstants::CellTracks* device_theCellTracksContainer_;

  unique_ptr<GPUCACell[]> device_theCells_;
  unique_ptr<GPUCACell::OuterHitOfCell[]> device_isOuterHitOfCell_;
  uint32_t* device_nCells_ = nullptr;

  unique_ptr<HitToTuple> device_hitToTuple_;
  unique_ptr<HitToTuple::Counter[]> device_hitToTupleStorage_;
  HitToTuple::View hitToTupleView_;

  cms::cuda::AtomicPairCounter* device_hitToTuple_apc_ = nullptr;

  cms::cuda::AtomicPairCounter* device_hitTuple_apc_ = nullptr;

  unique_ptr<TupleMultiplicity> device_tupleMultiplicity_;

  unique_ptr<cms::cuda::AtomicPairCounter::c_type[]> device_storage_;
  // params
  Params const& params_;
  /// Intermediate result avoiding repeated computations.
  const uint32_t paramsMaxDoubletes3Quarters_;
  /// Compute the number of doublet blocks for block size
  inline uint32_t nDoubletBlocks(uint32_t blockSize) {
    // We want (3 * params_.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize, but first part is pre-computed.
    return (paramsMaxDoubletes3Quarters_ + blockSize - 1) / blockSize;
  }

  /// Compute the number of quadruplet blocks for block size
  inline uint32_t nQuadrupletBlocks(uint32_t blockSize) {
    // caConstants::maxNumberOfQuadruplets is a constexpr, so the compiler will pre compute the 3*max/4
    return (3 * caConstants::maxNumberOfQuadruplets / 4 + blockSize - 1) / blockSize;
  }
};

using CAHitNtupletGeneratorKernelsGPU = CAHitNtupletGeneratorKernels<cms::cudacompat::GPUTraits>;
using CAHitNtupletGeneratorKernelsCPU = CAHitNtupletGeneratorKernels<cms::cudacompat::CPUTraits>;

#endif  // RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorKernels_h
