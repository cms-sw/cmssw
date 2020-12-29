#ifndef CalibTracker_SiPixelESProducers_SiPixelGainCalibrationForHLTGPU_H
#define CalibTracker_SiPixelESProducers_SiPixelGainCalibrationForHLTGPU_H

#include "HeterogeneousCore/CUDACore/interface/CUDAESProduct.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLT.h"

#include <cuda/api_wrappers.h>

class SiPixelGainCalibrationForHLT;
class SiPixelGainForHLTonGPU;
struct SiPixelGainForHLTonGPU_DecodingStructure;
class TrackerGeometry;

class SiPixelGainCalibrationForHLTGPU {
public:
  explicit SiPixelGainCalibrationForHLTGPU(const SiPixelGainCalibrationForHLT& gains, const TrackerGeometry& geom);
  ~SiPixelGainCalibrationForHLTGPU();

  const SiPixelGainForHLTonGPU *getGPUProductAsync(cuda::stream_t<>& cudaStream) const;

private:
  const SiPixelGainCalibrationForHLT *gains_ = nullptr;
  SiPixelGainForHLTonGPU *gainForHLTonHost_ = nullptr;
  struct GPUData {
    ~GPUData();
    SiPixelGainForHLTonGPU *gainForHLTonGPU = nullptr;
    SiPixelGainForHLTonGPU_DecodingStructure *gainDataOnGPU = nullptr;
  };
  CUDAESProduct<GPUData> gpuData_;
};

#endif
