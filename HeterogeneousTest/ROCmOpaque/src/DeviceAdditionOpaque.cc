#include <cstddef>

#include <hip/hip_runtime.h>

#include "HeterogeneousTest/ROCmWrapper/interface/DeviceAdditionWrapper.h"
#include "HeterogeneousTest/ROCmOpaque/interface/DeviceAdditionOpaque.h"
#include "HeterogeneousCore/ROCmUtilities/interface/hipCheck.h"

namespace cms::rocmtest {

  void opqaue_add_vectors_f(const float* in1_h, const float* in2_h, float* out_h, size_t size) {
    // allocate input and output buffers on the device
    float* in1_d;
    float* in2_d;
    float* out_d;
    hipCheck(hipMalloc(&in1_d, size * sizeof(float)));
    hipCheck(hipMalloc(&in2_d, size * sizeof(float)));
    hipCheck(hipMalloc(&out_d, size * sizeof(float)));

    // copy the input data to the device
    hipCheck(hipMemcpy(in1_d, in1_h, size * sizeof(float), hipMemcpyHostToDevice));
    hipCheck(hipMemcpy(in2_d, in2_h, size * sizeof(float), hipMemcpyHostToDevice));

    // fill the output buffer with zeros
    hipCheck(hipMemset(out_d, 0, size * sizeof(float)));

    // launch the 1-dimensional kernel for vector addition
    wrapper_add_vectors_f(in1_d, in2_d, out_d, size);

    // copy the results from the device to the host
    hipCheck(hipMemcpy(out_h, out_d, size * sizeof(float), hipMemcpyDeviceToHost));

    // wait for all the operations to complete
    hipCheck(hipDeviceSynchronize());

    // free the input and output buffers on the device
    hipCheck(hipFree(in1_d));
    hipCheck(hipFree(in2_d));
    hipCheck(hipFree(out_d));
  }

  void opqaue_add_vectors_d(const double* in1_h, const double* in2_h, double* out_h, size_t size) {
    // allocate input and output buffers on the device
    double* in1_d;
    double* in2_d;
    double* out_d;
    hipCheck(hipMalloc(&in1_d, size * sizeof(double)));
    hipCheck(hipMalloc(&in2_d, size * sizeof(double)));
    hipCheck(hipMalloc(&out_d, size * sizeof(double)));

    // copy the input data to the device
    hipCheck(hipMemcpy(in1_d, in1_h, size * sizeof(double), hipMemcpyHostToDevice));
    hipCheck(hipMemcpy(in2_d, in2_h, size * sizeof(double), hipMemcpyHostToDevice));

    // fill the output buffer with zeros
    hipCheck(hipMemset(out_d, 0, size * sizeof(double)));

    // launch the 1-dimensional kernel for vector addition
    wrapper_add_vectors_d(in1_d, in2_d, out_d, size);

    // copy the results from the device to the host
    hipCheck(hipMemcpy(out_h, out_d, size * sizeof(double), hipMemcpyDeviceToHost));

    // wait for all the operations to complete
    hipCheck(hipDeviceSynchronize());

    // free the input and output buffers on the device
    hipCheck(hipFree(in1_d));
    hipCheck(hipFree(in2_d));
    hipCheck(hipFree(out_d));
  }

}  // namespace cms::rocmtest
