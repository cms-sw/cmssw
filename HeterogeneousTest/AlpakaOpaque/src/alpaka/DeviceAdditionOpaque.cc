#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousTest/AlpakaOpaque/interface/alpaka/DeviceAdditionOpaque.h"
#include "HeterogeneousTest/AlpakaWrapper/interface/alpaka/DeviceAdditionWrapper.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::test {

  void opaque_add_vectors_f(const float* in1, const float* in2, float* out, uint32_t size) {
    // run on the first available devices
    auto const& device = cms::alpakatools::devices<Platform>()[0];
    Queue queue{device};

    // wrap the input and output data in views
    auto in1_h = cms::alpakatools::make_host_view<const float>(in1, size);
    auto in2_h = cms::alpakatools::make_host_view<const float>(in2, size);
    auto out_h = cms::alpakatools::make_host_view<float>(out, size);

    // allocate input and output buffers on the device
    auto in1_d = cms::alpakatools::make_device_buffer<float[]>(queue, size);
    auto in2_d = cms::alpakatools::make_device_buffer<float[]>(queue, size);
    auto out_d = cms::alpakatools::make_device_buffer<float[]>(queue, size);

    // copy the input data to the device
    // FIXME: pass the explicit size of type uint32_t to avoid compilation error
    // The destination view and the extent are required to have compatible index types!
    alpaka::memcpy(queue, in1_d, in1_h, size);
    alpaka::memcpy(queue, in2_d, in2_h, size);

    // fill the output buffer with zeros
    alpaka::memset(queue, out_d, 0);

    // launch the 1-dimensional kernel for vector addition
    test::wrapper_add_vectors_f(queue, in1_d.data(), in2_d.data(), out_d.data(), size);

    // copy the results from the device to the host
    alpaka::memcpy(queue, out_h, out_d);

    // wait for all the operations to complete
    alpaka::wait(queue);

    // the device buffers are freed automatically
  }

  void opaque_add_vectors_d(const double* in1, const double* in2, double* out, uint32_t size) {
    // run on the first available devices
    auto const& device = cms::alpakatools::devices<Platform>()[0];
    Queue queue{device};

    // wrap the input and output data in views
    auto in1_h = cms::alpakatools::make_host_view<const double>(in1, size);
    auto in2_h = cms::alpakatools::make_host_view<const double>(in2, size);
    auto out_h = cms::alpakatools::make_host_view<double>(out, size);

    // allocate input and output buffers on the device
    auto in1_d = cms::alpakatools::make_device_buffer<double[]>(queue, size);
    auto in2_d = cms::alpakatools::make_device_buffer<double[]>(queue, size);
    auto out_d = cms::alpakatools::make_device_buffer<double[]>(queue, size);

    // copy the input data to the device
    // FIXME: pass the explicit size of type uint32_t to avoid compilation error
    // The destination view and the extent are required to have compatible index types!
    alpaka::memcpy(queue, in1_d, in1_h, size);
    alpaka::memcpy(queue, in2_d, in2_h, size);

    // fill the output buffer with zeros
    alpaka::memset(queue, out_d, 0);

    // launch the 1-dimensional kernel for vector addition
    test::wrapper_add_vectors_d(queue, in1_d.data(), in2_d.data(), out_d.data(), size);

    // copy the results from the device to the host
    alpaka::memcpy(queue, out_h, out_d);

    // wait for all the operations to complete
    alpaka::wait(queue);

    // the device buffers are freed automatically
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::test
