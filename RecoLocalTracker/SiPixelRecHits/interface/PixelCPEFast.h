#ifndef RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h
#define RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h

#include <utility>

#include "CondFormats/SiPixelTransient/interface/SiPixelGenError.h"
#include "CondFormats/SiPixelTransient/interface/SiPixelTemplate.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEGenericBase.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

class MagneticField;
template <typename TrackerTraits>
class PixelCPEFast final : public PixelCPEGenericBase {
public:
  PixelCPEFast(edm::ParameterSet const &conf,
               const MagneticField *,
               const TrackerGeometry &,
               const TrackerTopology &,
               const SiPixelLorentzAngle *,
               const SiPixelGenErrorDBObject *,
               const SiPixelLorentzAngle *);

  ~PixelCPEFast() override = default;

  static void fillPSetDescription(edm::ParameterSetDescription &desc);

  // The return value can only be used safely in kernels launched on
  // the same cudaStream, or after cudaStreamSynchronize.
  using ParamsOnGPU = pixelCPEforGPU::ParamsOnGPUT<TrackerTraits>;
  using LayerGeometry = pixelCPEforGPU::LayerGeometryT<TrackerTraits>;
  using AverageGeometry = pixelTopology::AverageGeometryT<TrackerTraits>;

  const ParamsOnGPU *getGPUProductAsync(cudaStream_t cudaStream) const;

  ParamsOnGPU const &getCPUProduct() const { return cpuData_; }

private:
  LocalPoint localPosition(DetParam const &theDetParam, ClusterParam &theClusterParam) const override;
  LocalError localError(DetParam const &theDetParam, ClusterParam &theClusterParam) const override;

  void errorFromTemplates(DetParam const &theDetParam, ClusterParamGeneric &theClusterParam, float qclus) const;

  //--- DB Error Parametrization object, new light templates
  std::vector<SiPixelGenErrorStore> thePixelGenError_;

  // allocate this with posix malloc to be compatible with the cpu workflow
  std::vector<pixelCPEforGPU::DetParams> detParamsGPU_;
  pixelCPEforGPU::CommonParams commonParamsGPU_;
  LayerGeometry layerGeometry_;
  AverageGeometry averageGeometry_;
  ParamsOnGPU cpuData_;

  struct GPUData {
    ~GPUData();
    // not needed if not used on CPU...
    ParamsOnGPU paramsOnGPU_h;
    ParamsOnGPU *paramsOnGPU_d = nullptr;  // copy of the above on the Device
  };
  cms::cuda::ESProduct<GPUData> gpuData_;

  void fillParamsForGpu();
};

#endif  // RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h
