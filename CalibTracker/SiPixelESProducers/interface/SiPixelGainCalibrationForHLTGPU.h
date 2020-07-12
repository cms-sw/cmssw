#ifndef CalibTracker_SiPixelESProducers_interface_SiPixelGainCalibrationForHLTGPU_h
#define CalibTracker_SiPixelESProducers_interface_SiPixelGainCalibrationForHLTGPU_h

#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLT.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"

class SiPixelGainCalibrationForHLT;
class SiPixelGainForHLTonGPU;
struct SiPixelGainForHLTonGPU_DecodingStructure;
class TrackerGeometry;

class SiPixelGainCalibrationForHLTGPU {
public:
  explicit SiPixelGainCalibrationForHLTGPU(const SiPixelGainCalibrationForHLT &gains, const TrackerGeometry &geom);
  ~SiPixelGainCalibrationForHLTGPU();

  const SiPixelGainForHLTonGPU *getGPUProductAsync(cudaStream_t cudaStream) const;
  const SiPixelGainForHLTonGPU *getCPUProduct() const { return gainForHLTonHost_; }
  const SiPixelGainCalibrationForHLT *getOriginalProduct() { return gains_; }

private:
  const SiPixelGainCalibrationForHLT *gains_ = nullptr;
  SiPixelGainForHLTonGPU *gainForHLTonHost_ = nullptr;
  struct GPUData {
    ~GPUData();
    SiPixelGainForHLTonGPU *gainForHLTonGPU = nullptr;
    SiPixelGainForHLTonGPU_DecodingStructure *gainDataOnGPU = nullptr;
  };
  cms::cuda::ESProduct<GPUData> gpuData_;
};

#endif  // CalibTracker_SiPixelESProducers_interface_SiPixelGainCalibrationForHLTGPU_h
