#ifndef RecoPixelVertexing_PixelTriplets_plugins_CAHitQuadrupletGeneratorKernels_h
#define RecoPixelVertexing_PixelTriplets_plugins_CAHitQuadrupletGeneratorKernels_h

#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "RecoPixelVertexing/PixelTriplets/plugins/pixelTuplesHeterogeneousProduct.h"

#include "GPUCACell.h"

class CAHitQuadrupletGeneratorKernels {
public:
  // counters
  struct Counters {
    unsigned long long nEvents;
    unsigned long long nHits;
    unsigned long long nCells;
    unsigned long long nTuples;
    unsigned long long nGoodTracks;
    unsigned long long nUsedHits;
    unsigned long long nDupHits;
    unsigned long long nKilledCells;
    unsigned long long nEmptyCells;
    unsigned long long nZeroTrackCells;
  };

  using HitsOnGPU = TrackingRecHit2DSOAView;
  using HitsOnCPU = TrackingRecHit2DCUDA;

  using TuplesOnGPU = pixelTuplesHeterogeneousProduct::TuplesOnGPU;

  using HitToTuple = CAConstants::HitToTuple;
  using TupleMultiplicity = CAConstants::TupleMultiplicity;

  struct QualityCuts {
    // chi2 cut = chi2Scale * (chi2Coeff[0] + pT/GeV * (chi2Coeff[1] + pT/GeV * (chi2Coeff[2] + pT/GeV * chi2Coeff[3])))
    float chi2Coeff[4];
    float chi2MaxPt;    // GeV
    float chi2Scale;

    struct region {
      float maxTip;     // cm
      float minPt;      // GeV
      float maxZip;     // cm
    };

    region triplet;
    region quadruplet;
  };

  CAHitQuadrupletGeneratorKernels(uint32_t minHitsPerNtuplet,
                                  bool earlyFishbone,
                                  bool lateFishbone,
                                  bool idealConditions,
                                  bool doStats,
                                  bool doClusterCut,
                                  bool doZCut,
                                  bool doPhiCut,
                                  float ptmin,
                                  float CAThetaCutBarrel,
                                  float CAThetaCutForward,
                                  float hardCurvCut,
                                  float dcaCutInnerTriplet,
                                  float dcaCutOuterTriplet,
                                  QualityCuts const& cuts)
      : minHitsPerNtuplet_(minHitsPerNtuplet),
        earlyFishbone_(earlyFishbone),
        lateFishbone_(lateFishbone),
        idealConditions_(idealConditions),
        doStats_(doStats),
        doClusterCut_(doClusterCut),
        doZCut_(doZCut),
        doPhiCut_(doPhiCut),
        ptmin_(ptmin),
        CAThetaCutBarrel_(CAThetaCutBarrel),
        CAThetaCutForward_(CAThetaCutForward),
        hardCurvCut_(hardCurvCut),
        dcaCutInnerTriplet_(dcaCutInnerTriplet),
        dcaCutOuterTriplet_(dcaCutOuterTriplet),
        cuts_(cuts)
  { }

  ~CAHitQuadrupletGeneratorKernels() {
    deallocateOnGPU();
  }

  TupleMultiplicity const* tupleMultiplicity() const { return device_tupleMultiplicity_; }

  void launchKernels(HitsOnCPU const& hh, TuplesOnGPU& tuples_d, cudaStream_t cudaStream);

  void classifyTuples(HitsOnCPU const& hh, TuplesOnGPU& tuples_d, cudaStream_t cudaStream);

  void fillHitDetIndices(HitsOnCPU const &hh, TuplesOnGPU &tuples, TuplesOnGPU::Container * hitDetIndices, cuda::stream_t<>& stream);

  void buildDoublets(HitsOnCPU const& hh, cuda::stream_t<>& stream);
  void allocateOnGPU();
  void deallocateOnGPU();
  void cleanup(cudaStream_t cudaStream);
  void printCounters() const;

private:
  Counters* counters_ = nullptr;

  // workspace
  CAConstants::CellNeighborsVector* device_theCellNeighbors_ = nullptr;
  cudautils::device::unique_ptr<CAConstants::CellNeighbors[]> device_theCellNeighborsContainer_;
  CAConstants::CellTracksVector* device_theCellTracks_ = nullptr;
  cudautils::device::unique_ptr<CAConstants::CellTracks[]> device_theCellTracksContainer_;

  cudautils::device::unique_ptr<GPUCACell[]> device_theCells_;
  cudautils::device::unique_ptr<GPUCACell::OuterHitOfCell[]> device_isOuterHitOfCell_;
  uint32_t* device_nCells_ = nullptr;

  HitToTuple* device_hitToTuple_ = nullptr;
  AtomicPairCounter* device_hitToTuple_apc_ = nullptr;

  TupleMultiplicity* device_tupleMultiplicity_ = nullptr;
  uint8_t* device_tmws_ = nullptr;

  // params
  const uint32_t minHitsPerNtuplet_;
  const bool earlyFishbone_;
  const bool lateFishbone_;
  const bool idealConditions_;
  const bool doStats_;
  const bool doClusterCut_;
  const bool doZCut_;
  const bool doPhiCut_;
  const float ptmin_;
  const float CAThetaCutBarrel_;
  const float CAThetaCutForward_;
  const float hardCurvCut_;
  const float dcaCutInnerTriplet_;
  const float dcaCutOuterTriplet_;

  // quality cuts
  QualityCuts cuts_
  {
    // polynomial coefficients for the pT-dependent chi2 cut
    { 0.68177776, 0.74609577, -0.08035491, 0.00315399 },
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
};

#endif  // RecoPixelVertexing_PixelTriplets_plugins_CAHitQuadrupletGeneratorKernels_h
