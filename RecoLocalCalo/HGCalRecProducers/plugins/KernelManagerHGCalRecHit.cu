#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include "RecoLocalCalo/HGCalRecProducers/plugins/KernelManagerHGCalRecHit.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalRecHitKernelImpl.cuh"

KernelManagerHGCalRecHit::KernelManagerHGCalRecHit()
{
  ::nb_rechits_ = (data_->nhits_ + ::nt_rechits_.x - 1) / ::nt_rechits_.x;
}

KernelManagerHGCalRecHit::KernelManagerHGCalRecHit(KernelModifiableData<HGCUncalibratedRecHitSoA, HGCRecHitSoA> *data):
  data_(data)
{
  ::nb_rechits_ = (data_->nhits_ + ::nt_rechits_.x - 1) / ::nt_rechits_.x;
  nbytes_host_ = (data_->h_out_)->nbytes_ * data_->stride_;
  nbytes_device_ = (data_->d_1_)->nbytes_ * data_->stride_;
}

KernelManagerHGCalRecHit::~KernelManagerHGCalRecHit()
{
}

void KernelManagerHGCalRecHit::transfer_soas_to_device_(const cudaStream_t& stream)
{
  cudaCheck( cudaMemcpyAsync((data_->d_1_)->amplitude_, (data_->h_in_)->amplitude_, nbytes_device_, cudaMemcpyHostToDevice, stream) );
  cudaCheck( cudaGetLastError() );
}

void KernelManagerHGCalRecHit::transfer_soa_to_host_and_synchronize_(const cudaStream_t& stream)
{
  cudaCheck( cudaMemcpyAsync((data_->h_out_)->energy_, (data_->d_out_)->energy_, nbytes_host_, cudaMemcpyDeviceToHost, stream) );
  cudaCheck( cudaGetLastError() );
}

void KernelManagerHGCalRecHit::reuse_device_pointers_()
{
  std::swap(data_->d_1_, data_->d_2_); 
}

void KernelManagerHGCalRecHit::run_kernels(const KernelConstantData<HGCeeUncalibratedRecHitConstantData> *kcdata, const cudaStream_t& stream)
{
  transfer_soas_to_device_( stream );
  cudaCheck( cudaGetLastError() );
    
  /*
  ee_step1<<<::nb_rechits_, ::nt_rechits_, 0, stream>>>( *(data_->d_2_), *(data_->d_1_), kcdata->data_, data_->nhits_ );
  after_();
  reuse_device_pointers_();
  */

  ee_to_rechit<<<::nb_rechits_, ::nt_rechits_, 0, stream>>>( *(data_->d_out_), *(data_->d_1_), kcdata->data_, data_->nhits_ );
  cudaCheck( cudaGetLastError() );

  transfer_soa_to_host_and_synchronize_( stream );
  cudaCheck( cudaGetLastError() );
}

void KernelManagerHGCalRecHit::run_kernels(const KernelConstantData<HGChefUncalibratedRecHitConstantData> *kcdata, const cudaStream_t& stream)
{
  transfer_soas_to_device_( stream );
  cudaCheck( cudaGetLastError() );

  /*
  hef_step1<<<::nb_rechits_, ::nt_rechits_>>>( *(data_->d_2), *(data_->d_1_), d_kcdata->data, data_->nhits_);
  cudaCheck( cudaGetLastError() );
  reuse_device_pointers_();
  */

  hef_to_rechit<<<::nb_rechits_, ::nt_rechits_, 0, stream>>>( *(data_->d_out_), *(data_->d_1_), kcdata->data_, data_->nhits_ );
  cudaCheck( cudaGetLastError() );

  transfer_soa_to_host_and_synchronize_( stream );
  cudaCheck( cudaGetLastError() );
}

void KernelManagerHGCalRecHit::run_kernels(const KernelConstantData<HGChebUncalibratedRecHitConstantData> *kcdata, const cudaStream_t& stream)
{
  transfer_soas_to_device_( stream );
  cudaCheck( cudaGetLastError() );

  /*
  heb_step1<<<::nb_rechits_, nt_rechits_>>>( *(data_->d_2_), *(data_->d_1_), d_kcdata->data_, data_->nhits_);
  cudaCheck( cudaGetLastError() );
  reuse_device_pointers_();
  */

  heb_to_rechit<<<::nb_rechits_, ::nt_rechits_, 0, stream>>>( *(data_->d_out_), *(data_->d_1_), kcdata->data_, data_->nhits_ );
  cudaCheck( cudaGetLastError() );

  transfer_soa_to_host_and_synchronize_( stream );
  cudaCheck( cudaGetLastError() );
}

HGCRecHitSoA* KernelManagerHGCalRecHit::get_output()
{
  return data_->h_out_;
}
