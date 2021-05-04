#include <iostream>
#include <cassert>
#include <type_traits>

#include <cuda.h>
#include <cuda_runtime.h>

#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalRecHit/interface/HFQIE10Info.h"
#include "DataFormats/HcalRecHit/interface/HBHEChannelInfo.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"

template <typename T>
__global__ void kernel_test_hcal_rechits(T *other) {
  T rh(HcalDetId(0), 10.0f, 10.0f);
  other->setEnergy(rh.energy());
  other->setTime(rh.time());
}

__global__ void kernel_test_hcal_hfqie10info() { HFQIE10Info info; }

__global__ void kernel_test_hcal_hbhechinfo(HBHEChannelInfo *other) {
  HBHEChannelInfo info{true, true};
  info.setChannelInfo(HcalDetId{0}, 10, 10, 10, 1, 2.0, 2.0, 2.0, 0.0, false, false, false);
  other->setChannelInfo(info.id(),
                        info.recoShape(),
                        info.nSamples(),
                        info.soi(),
                        info.capid(),
                        info.darkCurrent(),
                        info.fcByPE(),
                        info.lambda(),
                        info.noisecorr(),
                        info.hasLinkError(),
                        info.hasCapidError(),
                        info.isDropped());
}

void test_hcal_hfqie10info() {
  auto check_error = [](auto code) {
    if (code != cudaSuccess) {
      std::cout << cudaGetErrorString(code) << std::endl;
      assert(false);
    }
  };

  kernel_test_hcal_hfqie10info<<<1, 1>>>();
  check_error(cudaGetLastError());
}

template <typename T>
void test_hcal_rechits() {
  auto check_error = [](auto code) {
    if (code != cudaSuccess) {
      std::cout << cudaGetErrorString(code) << std::endl;
      assert(false);
    }
  };

  T h_rh, h_rh_test{HcalDetId(0), 10.0f, 10.0f};
  T *d_rh;

  cudaMalloc((void **)&d_rh, sizeof(T));
  cudaMemcpy(d_rh, &h_rh, sizeof(T), cudaMemcpyHostToDevice);
  kernel_test_hcal_rechits<T><<<1, 1>>>(d_rh);
  cudaDeviceSynchronize();
  check_error(cudaGetLastError());
  cudaMemcpy(&h_rh, d_rh, sizeof(T), cudaMemcpyDeviceToHost);

  std::cout << h_rh << std::endl;
  std::cout << h_rh_test << std::endl;
  assert(h_rh.energy() == h_rh_test.energy());
  assert(h_rh.time() == h_rh_test.time());

  std::cout << "all good in " << __FUNCTION__ << std::endl;
}

void test_hcal_hbhechinfo() {
  auto check_error = [](auto code) {
    if (code != cudaSuccess) {
      std::cout << cudaGetErrorString(code) << std::endl;
      assert(false);
    }
  };

  HBHEChannelInfo h_info, h_info_test{true, true};
  h_info_test.setChannelInfo(HcalDetId{0}, 10, 10, 10, 1, 2.0, 2.0, 2.0, 0.0, false, false, false);
  HBHEChannelInfo *d_info;

  cudaMalloc((void **)&d_info, sizeof(HBHEChannelInfo));
  cudaMemcpy(d_info, &h_info, sizeof(HBHEChannelInfo), cudaMemcpyHostToDevice);
  kernel_test_hcal_hbhechinfo<<<1, 1>>>(d_info);
  cudaDeviceSynchronize();
  check_error(cudaGetLastError());
  cudaMemcpy(&h_info, d_info, sizeof(HBHEChannelInfo), cudaMemcpyDeviceToHost);

  assert(h_info.id() == h_info_test.id());
  assert(h_info.recoShape() == h_info_test.recoShape());
  assert(h_info.nSamples() == h_info_test.nSamples());
  assert(h_info.soi() == h_info_test.soi());
  assert(h_info.capid() == h_info_test.capid());
  assert(h_info.darkCurrent() == h_info_test.darkCurrent());
  assert(h_info.fcByPE() == h_info_test.fcByPE());
  assert(h_info.lambda() == h_info_test.lambda());
  assert(h_info.noisecorr() == h_info_test.noisecorr());
  assert(h_info.hasLinkError() == h_info_test.hasLinkError());
  assert(h_info.hasCapidError() == h_info_test.hasCapidError());

  std::cout << "all good in " << __FUNCTION__ << std::endl;
}

int main(int argc, char **argv) {
  cms::cudatest::requireDevices();

  test_hcal_rechits<HBHERecHit>();
  test_hcal_rechits<HFRecHit>();
  test_hcal_rechits<HORecHit>();
  test_hcal_hbhechinfo();

  std::cout << "all good" << std::endl;
  return 0;
}
