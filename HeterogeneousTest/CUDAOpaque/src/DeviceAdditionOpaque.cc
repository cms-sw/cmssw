#include <cstddef>

#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousTest/CUDAOpaque/interface/DeviceAdditionOpaque.h"
#include "HeterogeneousTest/CUDAWrapper/interface/DeviceAdditionWrapper.h"

namespace cms::cudatest {

  void opaque_add_vectors_f(const float* in1_h, const float* in2_h, float* out_h, size_t size) {
    // allocate input and output buffers on the device
    float* in1_d;
    float* in2_d;
    float* out_d;
    cudaCheck(cudaMalloc(&in1_d, size * sizeof(float)));
    cudaCheck(cudaMalloc(&in2_d, size * sizeof(float)));
    cudaCheck(cudaMalloc(&out_d, size * sizeof(float)));

    // copy the input data to the device
    cudaCheck(cudaMemcpy(in1_d, in1_h, size * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(in2_d, in2_h, size * sizeof(float), cudaMemcpyHostToDevice));

    // fill the output buffer with zeros
    cudaCheck(cudaMemset(out_d, 0, size * sizeof(float)));

    // launch the 1-dimensional kernel for vector addition
    wrapper_add_vectors_f(in1_d, in2_d, out_d, size);

    // copy the results from the device to the host
    cudaCheck(cudaMemcpy(out_h, out_d, size * sizeof(float), cudaMemcpyDeviceToHost));

    // wait for all the operations to complete
    cudaCheck(cudaDeviceSynchronize());

    // free the input and output buffers on the device
    cudaCheck(cudaFree(in1_d));
    cudaCheck(cudaFree(in2_d));
    cudaCheck(cudaFree(out_d));
  }

  void opaque_add_vectors_d(const double* in1_h, const double* in2_h, double* out_h, size_t size) {
    // allocate input and output buffers on the device
    double* in1_d;
    double* in2_d;
    double* out_d;
    cudaCheck(cudaMalloc(&in1_d, size * sizeof(double)));
    cudaCheck(cudaMalloc(&in2_d, size * sizeof(double)));
    cudaCheck(cudaMalloc(&out_d, size * sizeof(double)));

    // copy the input data to the device
    cudaCheck(cudaMemcpy(in1_d, in1_h, size * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(in2_d, in2_h, size * sizeof(double), cudaMemcpyHostToDevice));

    // fill the output buffer with zeros
    cudaCheck(cudaMemset(out_d, 0, size * sizeof(double)));

    // launch the 1-dimensional kernel for vector addition
    wrapper_add_vectors_d(in1_d, in2_d, out_d, size);

    // copy the results from the device to the host
    cudaCheck(cudaMemcpy(out_h, out_d, size * sizeof(double), cudaMemcpyDeviceToHost));

    // wait for all the operations to complete
    cudaCheck(cudaDeviceSynchronize());

    // free the input and output buffers on the device
    cudaCheck(cudaFree(in1_d));
    cudaCheck(cudaFree(in2_d));
    cudaCheck(cudaFree(out_d));
  }

}  // namespace cms::cudatest
