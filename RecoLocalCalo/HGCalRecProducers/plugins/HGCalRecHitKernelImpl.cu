#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "HGCalRecHitKernelImpl.cuh"

__device__ float get_weight_from_layer(const int32_t& layer,
                                       const double (&weights)[HGChefUncalibRecHitConstantData::hef_weights]) {
  return (float)weights[layer];
}

__device__ void make_rechit_silicon(unsigned tid,
                                    HGCRecHitSoA dst_soa,
                                    HGCUncalibRecHitSoA src_soa,
                                    const float& weight,
                                    const float& rcorr,
                                    const float& cce_correction,
                                    const float& sigmaNoiseGeV,
                                    const float& xmin,
                                    const float& xmax,
                                    const float& aterm,
                                    const float& cterm) {
  dst_soa.id_[tid] = src_soa.id_[tid];
  dst_soa.energy_[tid] = src_soa.amplitude_[tid] * weight * 0.001f * __fdividef(rcorr, cce_correction);
  dst_soa.time_[tid] = src_soa.jitter_[tid];

  HeterogeneousHGCSiliconDetId detid(src_soa.id_[tid]);
  dst_soa.flagBits_[tid] = 0 | (0x1 << HGCRecHit::kGood);
  float son = __fdividef(dst_soa.energy_[tid], sigmaNoiseGeV);
  float son_norm = fminf(32.f, son) / 32.f * ((1 << 8) - 1);
  long int son_round = lroundf(son_norm);
  //there is an extra 0.125 factor in HGCRecHit::signalOverSigmaNoise(), which should not affect CPU/GPU comparison
  dst_soa.son_[tid] = static_cast<uint8_t>(son_round);

  //get time resolution
  //https://github.com/cms-sw/cmssw/blob/master/RecoLocalCalo/HGCalRecProducers/src/ComputeClusterTime.cc#L50
  /*Maxmin trick to avoid conditions within the kernel (having xmin < xmax)
    3 possibilities: 1) xval -> xmin -> xmax
    2) xmin -> xval -> xmax
    3) xmin -> xmax -> xval
    The time error is calculated with the number in the middle.
  */
  float denominator = fminf(fmaxf(son, xmin), xmax);
  float div_ = __fdividef(aterm, denominator);
  dst_soa.timeError_[tid] = dst_soa.time_[tid] < 0 ? -1 : __fsqrt_rn(div_ * div_ + cterm * cterm);
  //if dst_soa.time_[tid] < 1 always, then the above conditional expression can be replaced by
  //dst_soa.timeError_[tid] = fminf( fmaxf( dst_soa.time_[tid]-1, -1 ), sqrt( div_*div_ + cterm*cterm ) )
  //which is *not* conditional, and thus potentially faster; compare to HGCalRecHitWorkerSimple.cc
}

__device__ void make_rechit_scintillator(
    unsigned tid, HGCRecHitSoA dst_soa, HGCUncalibRecHitSoA src_soa, const float& weight, const float& sigmaNoiseGeV) {
  dst_soa.id_[tid] = src_soa.id_[tid];
  dst_soa.energy_[tid] = src_soa.amplitude_[tid] * weight * 0.001f;
  dst_soa.time_[tid] = src_soa.jitter_[tid];

  HeterogeneousHGCScintillatorDetId detid(src_soa.id_[tid]);
  dst_soa.flagBits_[tid] = 0 | (0x1 << HGCRecHit::kGood);
  float son = __fdividef(dst_soa.energy_[tid], sigmaNoiseGeV);
  float son_norm = fminf(32.f, son) / 32.f * ((1 << 8) - 1);
  long int son_round = lroundf(son_norm);
  //there is an extra 0.125 factor in HGCRecHit::signalOverSigmaNoise(), which should not affect CPU/GPU comparison
  dst_soa.son_[tid] = static_cast<uint8_t>(son_round);
  dst_soa.timeError_[tid] = -1;
}

__device__ float get_thickness_correction(const int& type,
                                          const double (&rcorr)[HGChefUncalibRecHitConstantData::hef_rcorr]) {
  return __fdividef(1.f, (float)rcorr[type]);
}

__device__ float get_noise(const int& type, const double (&noise_fC)[HGChefUncalibRecHitConstantData::hef_noise_fC]) {
  return (float)noise_fC[type];
}

__device__ float get_cce_correction(const int& type, const double (&cce)[HGChefUncalibRecHitConstantData::hef_cce]) {
  return (float)cce[type];
}

__device__ float get_fCPerMIP(const int& type,
                              const double (&fCPerMIP)[HGChefUncalibRecHitConstantData::hef_fCPerMIP]) {
  return (float)fCPerMIP[type];
}

__global__ void ee_to_rechit(HGCRecHitSoA dst_soa,
                             HGCUncalibRecHitSoA src_soa,
                             const HGCeeUncalibRecHitConstantData cdata,
                             int length) {
  unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;

  for (unsigned i = tid; i < length; i += blockDim.x * gridDim.x) {
    HeterogeneousHGCSiliconDetId detid(src_soa.id_[i]);
    int32_t layer = detid.layer();
    float weight = get_weight_from_layer(layer, cdata.weights_);
    float rcorr = get_thickness_correction(detid.type(), cdata.rcorr_);
    float noise = get_noise(detid.type(), cdata.noise_fC_);
    float cce_correction = get_cce_correction(detid.type(), cdata.cce_);
    float fCPerMIP = get_fCPerMIP(detid.type(), cdata.fCPerMIP_);
    float sigmaNoiseGeV = 1e-3 * weight * rcorr * __fdividef(noise, fCPerMIP);
    make_rechit_silicon(i,
                        dst_soa,
                        src_soa,
                        weight,
                        rcorr,
                        cce_correction,
                        sigmaNoiseGeV,
                        cdata.xmin_,
                        cdata.xmax_,
                        cdata.aterm_,
                        cdata.cterm_);
  }
  __syncthreads();
}

__global__ void hef_to_rechit(HGCRecHitSoA dst_soa,
                              HGCUncalibRecHitSoA src_soa,
                              const HGChefUncalibRecHitConstantData cdata,
                              int length) {
  unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
  for (unsigned i = tid; i < length; i += blockDim.x * gridDim.x) {
    /*Uncomment the lines set to 1. as soon as those factors are centrally defined for the HSi.
	CUDADataFormats/HGCal/interface/HGCUncalibRecHitsToRecHitsConstants.h maxsizes_constanats will perhaps have to be changed (change some 3's to 6's) 
      */
    HeterogeneousHGCSiliconDetId detid(src_soa.id_[i]);
    int32_t layer = detid.layer() + cdata.layerOffset_;
    float weight = get_weight_from_layer(layer, cdata.weights_);
    float rcorr = 1.f;  //get_thickness_correction(detid.type(), cdata.rcorr_);
    float noise = get_noise(detid.type(), cdata.noise_fC_);
    float cce_correction = 1.f;  //get_cce_correction(detid.type(), cdata.cce_);
    float fCPerMIP = get_fCPerMIP(detid.type(), cdata.fCPerMIP_);
    float sigmaNoiseGeV = 1e-3 * weight * rcorr * __fdividef(noise, fCPerMIP);
    make_rechit_silicon(i,
                        dst_soa,
                        src_soa,
                        weight,
                        rcorr,
                        cce_correction,
                        sigmaNoiseGeV,
                        cdata.xmin_,
                        cdata.xmax_,
                        cdata.aterm_,
                        cdata.cterm_);
  }
  __syncthreads();
}

__global__ void heb_to_rechit(HGCRecHitSoA dst_soa,
                              HGCUncalibRecHitSoA src_soa,
                              const HGChebUncalibRecHitConstantData cdata,
                              int length) {
  unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;

  for (unsigned i = tid; i < length; i += blockDim.x * gridDim.x) {
    HeterogeneousHGCScintillatorDetId detid(src_soa.id_[i]);
    int32_t layer = detid.layer() + cdata.layerOffset_;
    float weight = get_weight_from_layer(layer, cdata.weights_);
    float noise = cdata.noise_MIP_;
    float sigmaNoiseGeV = 1e-3 * noise * weight;
    make_rechit_scintillator(i, dst_soa, src_soa, weight, sigmaNoiseGeV);
  }
  __syncthreads();
}
