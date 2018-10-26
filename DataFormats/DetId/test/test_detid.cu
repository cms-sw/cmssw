#include <cuda_runtime.h>
#include <cuda.h>

#include <iostream>
#include <assert.h>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

__global__ void test_gen_detid(DetId* id, uint32_t const rawid) {
    DetId did{rawid};
    *id = did;
}

void test_detid() {
    // test det ids
    DetId h_id, h_id_test{100};
    DetId h_test0{1};
    DetId *d_id;

    cudaMalloc((void**)&d_id, sizeof(DetId));
    cudaMemcpy(d_id, &h_id, sizeof(DetId), cudaMemcpyHostToDevice);
    test_gen_detid<<<1,1>>>(d_id, 100);
    cudaMemcpy(&h_id, d_id, sizeof(DetId), cudaMemcpyDeviceToHost);
    
    assert(h_id_test == h_id);
    assert(h_id != h_test0);
}

int main(int argc, char** argv) {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    std::cout << "nDevices = " << nDevices << std::endl;

    // test det id functionality
    if (nDevices > 0)
        test_detid();
}
