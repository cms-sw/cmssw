#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include "RecoLocalCalo/HGCalRecProducers/plugins/KernelManagerHGCalRecHit.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalRecHitKernelImpl.cuh"

/*
KernelManagerHGCalRecHit::KernelManagerHGCalRecHit(HGCUncalibratedRecHitSoA* uncalibSoA, HGCRecHitSoA* calibSoA):
  data_(data)
{
  ::nb_rechits_ = (data_->nhits_ + ::nt_rechits_.x - 1) / ::nt_rechits_.x;
  nbytes_host_ = (data_->h_out_)->nbytes_ * data_->stride_;
  nbytes_device_ = (data_->d_1_)->nbytes_ * data_->stride_;
}
*/

KernelManagerHGCalRecHit::KernelManagerHGCalRecHit(const unsigned& nelems,
						   HGCUncalibratedRecHitSoA* uncalibSoA, HGCRecHitSoA* calibSoA):
  uncalibSoA_(uncalibSoA), calibSoA_(calibSoA),
{
  nbytes_host_ = calibSoA->nbytes_ * nelems;
}

KernelManagerHGCalRecHit::~KernelManagerHGCalRecHit()
{
}

void KernelManagerHGCalRecHit::transfer_soas_to_device_(const cudaStream_t& stream)
{
  cudaCheck( cudaMemcpyAsync((data_->d_1_)->amplitude_, (data_->h_in_)->amplitude_, nbytes_device_, cudaMemcpyHostToDevice, stream) );
  cudaCheck( cudaGetLastError() );
}

void KernelManagerHGCalRecHit::transfer_soa_to_host_(const cudaStream_t& stream)
{
  cudaCheck( cudaMemcpyAsync((data_->h_out_)->energy_, (data_->d_out_)->energy_, nbytes_host_, cudaMemcpyDeviceToHost, stream) );
  cudaCheck( cudaGetLastError() );
}

void KernelManagerHGCalRecHit::run_kernels(const KernelConstantData<HGCeeUncalibratedRecHitConstantData> *kcdata, const cudaStream_t& stream)
{
  transfer_soas_to_device_( stream );
  
  ee_to_rechit<<<::nb_rechits_, ::nt_rechits_, 0, stream>>>( *(data_->d_out_), *(data_->d_1_), kcdata->data_, data_->nhits_ );
  cudaCheck( cudaGetLastError() );

  //transfer_soa_to_host_( stream );
}

void KernelManagerHGCalRecHit::run_kernels(const KernelConstantData<HGChefUncalibratedRecHitConstantData> *kcdata, const cudaStream_t& stream)
{
  transfer_soas_to_device_( stream );
  
  hef_to_rechit<<<::nb_rechits_, ::nt_rechits_, 0, stream>>>( *(data_->d_out_), *(data_->d_1_), kcdata->data_, data_->nhits_ );
  cudaCheck( cudaGetLastError() );

  //transfer_soa_to_host_( stream );
}

void KernelManagerHGCalRecHit::run_kernels(const KernelConstantData<HGChebUncalibratedRecHitConstantData> *kcdata, const cudaStream_t& stream)
{
  transfer_soas_to_device_( stream );

  heb_to_rechit<<<::nb_rechits_, ::nt_rechits_, 0, stream>>>( *(data_->d_out_), *(data_->d_1_), kcdata->data_, data_->nhits_ );
  cudaCheck( cudaGetLastError() );

  //transfer_soa_to_host_( stream );
}

HGCRecHitSoA* KernelManagerHGCalRecHit::get_output()
{
  return data_->h_out_;
}
