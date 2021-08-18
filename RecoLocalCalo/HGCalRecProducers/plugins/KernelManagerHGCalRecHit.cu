#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include "RecoLocalCalo/HGCalRecProducers/plugins/KernelManagerHGCalRecHit.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalRecHitKernelImpl.cuh"

namespace {  //kernel parameters
  dim3 nb_rechits_;
  constexpr dim3 nt_rechits_(1024);
}  // namespace

KernelManagerHGCalRecHit::KernelManagerHGCalRecHit(const HGCUncalibRecHitSoA& h_uncalibSoA,
                                                   const HGCUncalibRecHitSoA& d_uncalibSoA,
                                                   const HGCRecHitSoA& d_calibSoA)
    : h_uncalibSoA_(h_uncalibSoA), d_uncalibSoA_(d_uncalibSoA), d_calibSoA_(d_calibSoA) {
  nhits_ = h_uncalibSoA_.nhits_;
  pad_ = h_uncalibSoA_.pad_;
  ::nb_rechits_ = (pad_ + ::nt_rechits_.x - 1) / ::nt_rechits_.x;
  nbytes_device_ = d_uncalibSoA_.nbytes_ * pad_;
}

KernelManagerHGCalRecHit::KernelManagerHGCalRecHit(const HGCRecHitSoA& h_calibSoA_,
                                                   const ConstHGCRecHitSoA& d_calibConstSoA)
    : h_calibSoA_(h_calibSoA_), d_calibConstSoA_(d_calibConstSoA) {
  nhits_ = h_calibSoA_.nhits_;
  pad_ = h_calibSoA_.pad_;
  ::nb_rechits_ = (pad_ + ::nt_rechits_.x - 1) / ::nt_rechits_.x;
  nbytes_host_ = h_calibSoA_.nbytes_ * pad_;
}

KernelManagerHGCalRecHit::~KernelManagerHGCalRecHit() {}

void KernelManagerHGCalRecHit::transfer_soa_to_device_(const cudaStream_t& stream) {
  cudaCheck(cudaMemcpyAsync(
      d_uncalibSoA_.amplitude_, h_uncalibSoA_.amplitude_, nbytes_device_, cudaMemcpyHostToDevice, stream));
}

void KernelManagerHGCalRecHit::transfer_soa_to_host(const cudaStream_t& stream) {
  cudaCheck(
      cudaMemcpyAsync(h_calibSoA_.energy_, d_calibConstSoA_.energy_, nbytes_host_, cudaMemcpyDeviceToHost, stream));
}

void KernelManagerHGCalRecHit::run_kernels(const KernelConstantData<HGCeeUncalibRecHitConstantData>* kcdata,
                                           const cudaStream_t& stream) {
  transfer_soa_to_device_(stream);
  ee_to_rechit<<<::nb_rechits_, ::nt_rechits_, 0, stream>>>(d_calibSoA_, d_uncalibSoA_, kcdata->data_, nhits_);
  cudaCheck(cudaGetLastError());
}

void KernelManagerHGCalRecHit::run_kernels(const KernelConstantData<HGChefUncalibRecHitConstantData>* kcdata,
                                           const cudaStream_t& stream) {
  transfer_soa_to_device_(stream);
  hef_to_rechit<<<::nb_rechits_, ::nt_rechits_, 0, stream>>>(d_calibSoA_, d_uncalibSoA_, kcdata->data_, nhits_);
  cudaCheck(cudaGetLastError());
}

void KernelManagerHGCalRecHit::run_kernels(const KernelConstantData<HGChebUncalibRecHitConstantData>* kcdata,
                                           const cudaStream_t& stream) {
  transfer_soa_to_device_(stream);
  heb_to_rechit<<<::nb_rechits_, ::nt_rechits_, 0, stream>>>(d_calibSoA_, d_uncalibSoA_, kcdata->data_, nhits_);
  cudaCheck(cudaGetLastError());
}
