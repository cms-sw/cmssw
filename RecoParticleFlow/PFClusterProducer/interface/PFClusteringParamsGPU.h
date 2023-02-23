#ifndef RecoParticleFlow_PFClusterProducer_PFClusteringParamsGPU_h
#define RecoParticleFlow_PFClusterProducer_PFClusteringParamsGPU_h

#include <vector>

#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Utilities/interface/propagate_const_array.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"
#include "CUDADataFormats/Common/interface/PortableHostCollection.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif

namespace edm {
  class ParameterSet;
  class ParameterSetDescription;
}  // namespace edm

class PFClusteringParamsGPU {
public:
  GENERATE_SOA_LAYOUT(ProductSoALayout,
                      SOA_SCALAR(int32_t, nNeigh),
                      SOA_SCALAR(float, seedPt2ThresholdEB),
                      SOA_SCALAR(float, seedPt2ThresholdEE),
                      SOA_COLUMN(float, seedEThresholdEB_vec),
                      SOA_COLUMN(float, seedEThresholdEE_vec),
                      SOA_COLUMN(float, topoEThresholdEB_vec),
                      SOA_COLUMN(float, topoEThresholdEE_vec),
                      SOA_SCALAR(float, showerSigma2),
                      SOA_SCALAR(float, minFracToKeep),
                      SOA_SCALAR(float, minFracTot),
                      SOA_SCALAR(uint32_t, maxIterations),
                      SOA_SCALAR(bool, excludeOtherSeeds),
                      SOA_SCALAR(float, stoppingTolerance),
                      SOA_SCALAR(float, minFracInCalc),
                      SOA_SCALAR(float, minAllowedNormalization),
                      SOA_COLUMN(float, recHitEnergyNormInvEB_vec),
                      SOA_COLUMN(float, recHitEnergyNormInvEE_vec),
                      SOA_SCALAR(float, barrelTimeResConsts_corrTermLowE),
                      SOA_SCALAR(float, barrelTimeResConsts_threshLowE),
                      SOA_SCALAR(float, barrelTimeResConsts_noiseTerm),
                      SOA_SCALAR(float, barrelTimeResConsts_constantTermLowE2),
                      SOA_SCALAR(float, barrelTimeResConsts_noiseTermLowE),
                      SOA_SCALAR(float, barrelTimeResConsts_threshHighE),
                      SOA_SCALAR(float, barrelTimeResConsts_constantTerm2),
                      SOA_SCALAR(float, barrelTimeResConsts_resHighE2),
                      SOA_SCALAR(float, endcapTimeResConsts_corrTermLowE),
                      SOA_SCALAR(float, endcapTimeResConsts_threshLowE),
                      SOA_SCALAR(float, endcapTimeResConsts_noiseTerm),
                      SOA_SCALAR(float, endcapTimeResConsts_constantTermLowE2),
                      SOA_SCALAR(float, endcapTimeResConsts_noiseTermLowE),
                      SOA_SCALAR(float, endcapTimeResConsts_threshHighE),
                      SOA_SCALAR(float, endcapTimeResConsts_constantTerm2),
                      SOA_SCALAR(float, endcapTimeResConsts_resHighE2))

  using HostProduct = cms::cuda::PortableHostCollection<ProductSoALayout<>>;
  using DeviceProduct = cms::cuda::PortableDeviceCollection<ProductSoALayout<>>;

#ifndef __CUDACC__
  PFClusteringParamsGPU(edm::ParameterSet const&);
  ~PFClusteringParamsGPU() = default;

  static void fillDescription(edm::ParameterSetDescription& psetDesc);

  DeviceProduct const& getProduct(cudaStream_t) const;

private:
  constexpr static uint32_t kMaxDepth_barrel = 4;
  constexpr static uint32_t kMaxDepth_endcap = 7;

  void setParameterValues(edm::ParameterSet const& iConfig);

  HostProduct params_;
  cms::cuda::ESProduct<DeviceProduct> product_;
#endif
};

#endif  // RecoParticleFlow_PFClusterProducer_PFClusteringParamsGPU_h
