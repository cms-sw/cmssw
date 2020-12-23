#ifndef CondFormats_SiPixelObjects_interface_SiPixelGainForHLTonGPU_h
#define CondFormats_SiPixelObjects_interface_SiPixelGainForHLTonGPU_h

#include <cstdint>
#include <cstdio>
#include <tuple>

// including <cuda_runtime.h> would pull in the dependency on all of CUDA;
// instead, just define away the CUDA specific attributes to keep GCC happy.
#ifndef __CUDACC__
#ifndef __host__
#define __host__
#endif  // __host__
#ifndef __device__
#define __device__
#endif  // __device__
#endif  // __CUDACC__

#include "CUDADataFormats/SiPixelCluster/interface/gpuClusteringConstants.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"

struct SiPixelGainForHLTonGPU_DecodingStructure {
  uint8_t gain;
  uint8_t ped;
};

// copy of SiPixelGainCalibrationForHLT
class SiPixelGainForHLTonGPU {
public:
  using DecodingStructure = SiPixelGainForHLTonGPU_DecodingStructure;

  using Range = std::pair<uint32_t, uint32_t>;

  inline __host__ __device__ std::pair<float, float> getPedAndGain(
      uint32_t moduleInd, int col, int row, bool& isDeadColumn, bool& isNoisyColumn) const {
    auto range = rangeAndCols_[moduleInd].first;
    auto nCols = rangeAndCols_[moduleInd].second;

    // determine what averaged data block we are in (there should be 1 or 2 of these depending on if plaquette is 1 by X or 2 by X
    unsigned int lengthOfColumnData = (range.second - range.first) / nCols;
    unsigned int lengthOfAveragedDataInEachColumn = 2;  // we always only have two values per column averaged block
    unsigned int numberOfDataBlocksToSkip = row / numberOfRowsAveragedOver_;

    auto offset = range.first + col * lengthOfColumnData + lengthOfAveragedDataInEachColumn * numberOfDataBlocksToSkip;

    assert(offset < range.second);
    assert(offset < 3088384);
    assert(0 == offset % 2);

    DecodingStructure const* __restrict__ lp = v_pedestals_;
    auto s = lp[offset / 2];

    isDeadColumn = (s.ped & 0xFF) == deadFlag_;
    isNoisyColumn = (s.ped & 0xFF) == noisyFlag_;

    return std::make_pair(decodePed(s.ped & 0xFF), decodeGain(s.gain & 0xFF));
  }

  constexpr float decodeGain(unsigned int gain) const { return gain * gainPrecision_ + minGain_; }
  constexpr float decodePed(unsigned int ped) const { return ped * pedPrecision_ + minPed_; }

  DecodingStructure* v_pedestals_;
  std::pair<Range, int> rangeAndCols_[gpuClustering::maxNumModules];

  float minPed_, maxPed_, minGain_, maxGain_;
  float pedPrecision_, gainPrecision_;

  unsigned int numberOfRowsAveragedOver_;  // this is 80!!!!
  unsigned int nBinsToUseForEncoding_;
  unsigned int deadFlag_;
  unsigned int noisyFlag_;
};

#endif  // CondFormats_SiPixelObjects_interface_SiPixelGainForHLTonGPU_h
