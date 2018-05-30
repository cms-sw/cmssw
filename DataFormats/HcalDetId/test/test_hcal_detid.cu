#include <cuda_runtime.h>
#include <cuda.h>

#include <iostream>
#include <assert.h>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

__global__ void test_gen_detid(DetId* id) {
    DetId did;
    *id = did;
}

__global__ void test_gen_hcal_detid(HcalDetId *id) {
    HcalDetId did(HcalBarrel, 5, 5, 0);
    *id = did;

    // trigger functions on the device
    did.iphi();
    did.ieta();
    did.zside();
    did.subdet();
    did.ietaAbs();
    did.depth();
    did.hfdepth();
    did.maskDepth();
    did.baseDetId();
    did.secondAnodeId();
    did.crystal_ieta_low();
    did.crystal_ieta_high();
    did.crystal_iphi_low();
    did.crystal_iphi_high();
}

void test_detid() {
    // test det ids
    DetId h_id, h_id_test;
    DetId h_test0{1};
    DetId *d_id;

    cudaMalloc((void**)&d_id, sizeof(DetId));
    cudaMemcpy(d_id, &h_id, sizeof(DetId), cudaMemcpyHostToDevice);
    test_gen_detid<<<1,1>>>(d_id);
    cudaMemcpy(&h_id_test, d_id, sizeof(DetId), cudaMemcpyDeviceToHost);
    
    assert(h_id_test == h_id);
    assert(h_id != h_test0);
}

void test_hcal_detid() {
    HcalDetId h_id;
    HcalDetId h_id_test0{HcalBarrel, 5, 5, 0};
    HcalDetId *d_id;

    cudaMalloc((void**)&d_id, sizeof(HcalDetId));
    cudaMemcpy(d_id, &h_id, sizeof(HcalDetId), cudaMemcpyHostToDevice);
    test_gen_hcal_detid<<<1,1>>>(d_id);
    cudaMemcpy(&h_id, d_id, sizeof(HcalDetId), cudaMemcpyDeviceToHost);

    std::cout << h_id_test0 << std::endl;
    std::cout << h_id << std::endl;
    assert(h_id_test0 == h_id);
}

int main(int argc, char** argv) {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    std::cout << "nDevices = " << nDevices << std::endl;

    // test det id functionality
    if (nDevices>0)
        test_detid();

    // test hcal det ids
    if (nDevices>0)
        test_hcal_detid();

    return 0;
}
