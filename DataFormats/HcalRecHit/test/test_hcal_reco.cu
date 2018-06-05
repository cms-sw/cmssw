#include <iostream>
#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>

#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"

__global__ void kernel_test_hcal_rechits(HBHERecHit* other) {
    HBHERecHit rh(HcalDetId(0), 10.0f, 10.0f, 10.0f);
    other->setEnergy(rh.energy());
    other->setTime(rh.time());
    other->setTimeFalling(rh.timeFalling());
}

void test_hcal_rechits() {
    auto check_error = [](auto code) {
        if (code != cudaSuccess)
            std::cout << cudaGetErrorString(code) << std::endl;
    };

    HBHERecHit h_rh, h_rh_test{HcalDetId(0), 10.0f, 10.0f, 10.0f};
    HBHERecHit *d_rh;

    cudaMalloc((void**)&d_rh, sizeof(HBHERecHit));
    cudaMemcpy(d_rh, &h_rh, sizeof(HBHERecHit), cudaMemcpyHostToDevice);
    kernel_test_hcal_rechits<<<1,1>>>(d_rh);
    cudaDeviceSynchronize();
    check_error(cudaGetLastError());
    cudaMemcpy(&h_rh, d_rh, sizeof(HBHERecHit), cudaMemcpyDeviceToHost);

    std::cout << h_rh << std::endl;
    std::cout << h_rh_test << std::endl;
    assert(h_rh.energy() == h_rh_test.energy());
    assert(h_rh.time() == h_rh_test.time());
    assert(h_rh.timeFalling() == h_rh_test.timeFalling());
    assert(h_rh.chi2() == h_rh_test.chi2());
}

int main(int argc, char ** argv) {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    std::cout << "nDevices = " << nDevices << std::endl;

    if (nDevices > 0) {
        test_hcal_rechits();

        std::cout << "all good" << std::endl;
    }

    return 0;
}
