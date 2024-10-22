#include <cassert>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"

__global__ void kernel_test_calo_rechit(CaloRecHit* other) {
  CaloRecHit rh{DetId(0), 10, 1, 0, 0};
  other->setEnergy(rh.energy());
  other->setTime(rh.time());
  other->setFlagField(10, 31, 1);
}

void test_calo_rechit() {
  auto check_error = [](auto code) {
    if (code != cudaSuccess) {
      std::cout << cudaGetErrorString(code) << std::endl;
      assert(false);
    }
  };

  CaloRecHit h_rh, h_rh_test{DetId(0), 10, 1, 0, 0};
  h_rh_test.setFlagField(10, 31, 1);
  CaloRecHit* d_rh;

  cudaMalloc((void**)&d_rh, sizeof(CaloRecHit));
  cudaMemcpy(d_rh, &h_rh, sizeof(CaloRecHit), cudaMemcpyHostToDevice);
  kernel_test_calo_rechit<<<1, 1>>>(d_rh);
  cudaDeviceSynchronize();
  check_error(cudaGetLastError());
  cudaMemcpy(&h_rh, d_rh, sizeof(CaloRecHit), cudaMemcpyDeviceToHost);

  std::cout << h_rh << std::endl;
  std::cout << h_rh_test << std::endl;
  assert(h_rh.energy() == h_rh_test.energy());
  assert(h_rh.time() == h_rh_test.time());
  assert(h_rh.flags() == h_rh_test.flags());
  assert(h_rh.aux() == h_rh_test.aux());
  assert(h_rh.detid() == h_rh_test.detid());
}

int main(int argc, char** argv) {
  cms::cudatest::requireDevices();

  test_calo_rechit();

  std::cout << "all good!" << std::endl;
  return 0;
}
