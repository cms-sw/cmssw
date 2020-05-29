#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "HGCalRecHitKernelImpl.cuh"

__device__ 
double get_weight_from_layer(const int& layer, const double (&weights)[maxsizes_constants::hef_weights])
{
  return weights[layer];
}

__device__
void make_rechit(unsigned int tid, HGCRecHitSoA& dst_soa, HGCUncalibratedRecHitSoA& src_soa, const bool &heb_flag, 
		 const double& weight, const double& rcorr, const double& cce_correction, const double &sigmaNoiseGeV,
		 const float& xmin, const float& xmax, const float& aterm, const float& cterm)
{
  dst_soa.id_[tid] = src_soa.id_[tid];
  dst_soa.energy_[tid] = src_soa.amplitude_[tid] * weight * 0.001f;
  if(!heb_flag)
    dst_soa.energy_[tid] *=  __fdividef(rcorr, cce_correction);
  dst_soa.time_[tid] = src_soa.jitter_[tid];
  dst_soa.flagBits_[tid] |= (0x1 << HGCRecHit::kGood);
  float son = __fdividef( dst_soa.energy_[tid], sigmaNoiseGeV);
  float son_norm = fminf(32.f, son) / 32.f * ((1 << 8)-1);
  long int son_round = lroundf( son_norm );
  dst_soa.son_[tid] = static_cast<uint8_t>( son_round );

  if(heb_flag==0)
    {
      //get time resolution
      //https://github.com/cms-sw/cmssw/blob/master/RecoLocalCalo/HGCalRecProducers/src/ComputeClusterTime.cc#L50
      /*Maxmin trick to avoid conditions within the kernel (having xmin < xmax)
      3 possibilities: 1) xval -> xmin -> xmax
                       2) xmin -> xval -> xmax
                       3) xmin -> xmax -> xval
      The time error is calculated with the number in the middle.
      */
      float max = fminf( fmaxf(son, xmin), xmax);
      float div_ = __fdividef(aterm, max);
      dst_soa.timeError_[tid] = sqrt( div_*div_ + cterm*cterm );
    }
  else
    dst_soa.timeError_[tid] = -1;
}

__device__ 
double get_thickness_correction(const int& type, const double (&rcorr)[maxsizes_constants::hef_rcorr])
{
  return rcorr[type];
}

__device__
double get_noise(const int& type, const double (&noise_fC)[maxsizes_constants::hef_noise_fC])
{
  return noise_fC[type - 1];
}

__device__
double get_cce_correction(const int& type, const double (&cce)[maxsizes_constants::hef_cce])
{
  return cce[type - 1];
}

__device__ 
double get_fCPerMIP(const int& type, const double (&fCPerMIP)[maxsizes_constants::hef_fCPerMIP])
{
  return fCPerMIP[type - 1];
}

__global__
void ee_step1(HGCUncalibratedRecHitSoA dst_soa, HGCUncalibratedRecHitSoA src_soa, const HGCeeUncalibratedRecHitConstantData cdata, int length)
{
}

__global__
void hef_step1(HGCUncalibratedRecHitSoA dst_soa, HGCUncalibratedRecHitSoA src_soa, const HGChefUncalibratedRecHitConstantData cdata, int length)
{
}

__global__
void heb_step1(HGCUncalibratedRecHitSoA dst_soa, HGCUncalibratedRecHitSoA src_soa, const HGChebUncalibratedRecHitConstantData cdata, int length)
{
}

__global__
void ee_to_rechit(HGCRecHitSoA dst_soa, HGCUncalibratedRecHitSoA src_soa, const HGCeeUncalibratedRecHitConstantData cdata, int length)
{
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  HeterogeneousHGCSiliconDetId detid(src_soa.id_[tid]);

  for (unsigned int i = tid; i < length; i += blockDim.x * gridDim.x)
    {
      double weight         = get_weight_from_layer(detid.layer(), cdata.weights_);
      double rcorr          = get_thickness_correction(detid.type(), cdata.rcorr_);
      double noise          = get_noise(detid.type(), cdata.noise_fC_);
      double cce_correction = get_cce_correction(detid.type(), cdata.cce_);
      double fCPerMIP       = get_fCPerMIP(detid.type(), cdata.fCPerMIP_);
      double sigmaNoiseGeV  = 1e-3 * weight * rcorr * __fdividef( noise,  fCPerMIP );
      make_rechit(i, dst_soa, src_soa, false, weight, rcorr, cce_correction, sigmaNoiseGeV,
		  cdata.xmin_, cdata.xmax_, cdata.aterm_, cdata.cterm_);
    }
}

__global__
void hef_to_rechit(HGCRecHitSoA dst_soa, HGCUncalibratedRecHitSoA src_soa, const HGChefUncalibratedRecHitConstantData cdata, const hgcal_conditions::HeterogeneousHEFConditionsESProduct* conds, int length)
{
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  HeterogeneousHGCSiliconDetId detid(src_soa.id_[tid]);
  printf("waferTypeL: %d - cellCoarseY: %lf - cellX: %d\n", conds->params.waferTypeL_[0], conds->params.cellCoarseY_[12], detid.cellX());

  for (unsigned int i = tid; i < length; i += blockDim.x * gridDim.x)
    {
      double weight         = get_weight_from_layer(detid.layer(), cdata.weights_);
      double rcorr          = get_thickness_correction(detid.type(), cdata.rcorr_);
      double noise          = get_noise(detid.type(), cdata.noise_fC_);
      double cce_correction = get_cce_correction(detid.type(), cdata.cce_);
      double fCPerMIP       = get_fCPerMIP(detid.type(), cdata.fCPerMIP_);
      double sigmaNoiseGeV  = 1e-3 * weight * rcorr * __fdividef( noise,  fCPerMIP );
      make_rechit(i, dst_soa, src_soa, false, weight, rcorr, cce_correction, sigmaNoiseGeV,
		  cdata.xmin_, cdata.xmax_, cdata.aterm_, cdata.cterm_);
    }
}

__global__
void heb_to_rechit(HGCRecHitSoA dst_soa, HGCUncalibratedRecHitSoA src_soa, const HGChebUncalibratedRecHitConstantData cdata, int length)
{
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  HeterogeneousHGCScintillatorDetId detid(src_soa.id_[tid]);

  for (unsigned int i = tid; i < length; i += blockDim.x * gridDim.x)
    {
      double weight        = get_weight_from_layer(detid.layer(), cdata.weights_);
      double noise         = cdata.noise_MIP_;
      double sigmaNoiseGeV = 1e-3 * noise * weight;
      make_rechit(i, dst_soa, src_soa, true, weight, 0., 0., sigmaNoiseGeV,
		  0, 0, 0, 0);
    }
}
