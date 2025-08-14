#include <cassert>
#include <cstdlib>
#include <iostream>
//#include <new>

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaMath/interface/Vector.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

using namespace cms::alpakatools::math;
using namespace cms::alpakatools;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;

using Vec3d    = cms::alpakatools::math::Vector<double, 3>;
using data_t   = typename Vec3d::value_type;
using scalar_t = data_t; 

constexpr int len{1 << 12};
constexpr int nSrc{Vec3d::N};
constexpr scalar_t s{3.14};

struct scaleKernel {
  const scalar_t a;

  constexpr scaleKernel() : a(s) {}
 
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, const data_t* x, data_t* z) const {
    for (auto i : uniform_elements(acc)) {
      const Vec3d xvec(x[i], x[i+len], x[i+len*2]);

      const Vec3d zvec = cms::alpakatools::math::scale(a, xvec);

      CMS_UNROLL_LOOP
      for(int n = 0; n < nSrc; n++) {
        z[i+n*len] = zvec[n];
      }
    }
  }
};

struct addKernel {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, const data_t* x ,const data_t* y, data_t* z) const {
    for (auto i : uniform_elements(acc)) {
      const Vec3d xvec(x[i], x[i+len], x[i+len*2]);
      const Vec3d yvec(y[i], y[i+len], y[i+len*2]);

      const Vec3d zvec = cms::alpakatools::math::add(xvec, yvec);

      CMS_UNROLL_LOOP
      for(int n = 0; n < nSrc; n++) {	      
        z[i+n*len] = zvec[n];
      }
    }
  }
};

struct subKernel {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, const data_t* x ,const data_t* y, data_t* z) const {
    for (auto i : uniform_elements(acc)){
      const Vec3d xvec(x[i], x[i+len], x[i+len*2]);
      const Vec3d yvec(y[i], y[i+len], y[i+len*2]);

      const Vec3d zvec = cms::alpakatools::math::sub(xvec, yvec);

      CMS_UNROLL_LOOP
      for(int n = 0; n < nSrc; n++) {
        z[i+n*len] = zvec[n];
      }
    }
  }
};


struct axpyKernel {
  const scalar_t a;

  constexpr axpyKernel() : a(s) {}

  ALPAKA_FN_ACC void operator()(Acc1D const& acc, const data_t* x , const data_t* y, data_t* z) const {
    for (auto i : uniform_elements(acc)){
      const Vec3d xvec(x[i], x[i+len], x[i+len*2]);
      const Vec3d yvec(y[i], y[i+len], y[i+len*2]);

      const Vec3d zvec = cms::alpakatools::math::axpy(a, xvec, yvec);

      CMS_UNROLL_LOOP
      for(int n = 0; n < nSrc; n++) {
        z[i+n*len] = zvec[n];
      }
    }
  }
};


struct normalizeKernel {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, data_t* x) const {
    for (auto i : uniform_elements(acc)){
      Vec3d xvec(x[i], x[i+len], x[i+len*2]);

      xvec.normalize(acc);

      CMS_UNROLL_LOOP
      for(int n = 0; n < nSrc; n++) {
        x[i+n*len] = xvec[n];
      }
    }
  }
};


struct normKernel {

  ALPAKA_FN_ACC void operator()(Acc1D const& acc, const data_t* x, data_t* r) const {
    for (auto i : uniform_elements(acc)){
      const Vec3d xvec(x[i], x[i+len], x[i+len*2]);

      r[i] = xvec.norm(acc);
    }
  }
};


struct partialNormKernel {

  ALPAKA_FN_ACC void operator()(Acc1D const& acc, const data_t* x, data_t* r) const {
    for (auto i : uniform_elements(acc)){
      const Vec3d xvec(x[i], x[i+len], x[i+len*2]);

      r[i] = xvec.template partial_norm<Acc1D, 2>(acc);
    }
  }
};

struct dotKernel {

  ALPAKA_FN_ACC void operator()(Acc1D const& acc, const data_t* x, const data_t* y, data_t* r) const {
    for (auto i : uniform_elements(acc)){
      const Vec3d xvec(x[i], x[i+len], x[i+len*2]);
      const Vec3d yvec(y[i], y[i+len], y[i+len*2]);

      r[i] = cms::alpakatools::math::dot(xvec, yvec);
    }
  }
};


int main() {
  // get the list of devices on the current platform
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cerr << "No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
      "the test will be skipped.\n";
    exit(EXIT_FAILURE);
  }

  constexpr int N = len;

  // run the test on each device
  for (auto const& device : devices) {
    
    auto queue = Queue(device);
    auto const host_platform = alpaka::PlatformCpu{};
    auto const host_device   = alpaka::getDevByIdx(host_platform, 0);

    auto x_h = make_host_buffer<data_t[]>(queue, N * nSrc);
    auto x_d = make_device_buffer<data_t[]>(queue, N * nSrc);

    auto y_h = make_host_buffer<data_t[]>(queue, N * nSrc);
    auto y_d = make_device_buffer<data_t[]>(queue, N * nSrc);

    auto z_h = make_host_buffer<data_t[]>(queue, N * nSrc);
    auto z_d = make_device_buffer<data_t[]>(queue, N * nSrc);

    auto result_h = make_host_buffer<data_t[]>(queue, N);
    auto result_d = make_device_buffer<data_t[]>(queue, N);

    Vec1D const extent1D(N * nSrc);

    auto x_view = alpaka::createView(host_device, x_h.data(), extent1D);
    auto y_view = alpaka::createView(host_device, y_h.data(), extent1D);
    auto z_view = alpaka::createView(host_device, z_h.data(), extent1D);

    auto result_view = alpaka::createView(host_device, result_h.data(), extent1D);

    // Initialize random number generator
    std::random_device rd; 
    std::mt19937 gen(rd()); 
    std::normal_distribution<data_t> distr(0.0, 1.0); 

    for(int i = 0; i < N * nSrc; i++) {
      x_view[i] = distr(gen);
      y_view[i] = distr(gen);
    }

    // copy the object to the device.
    alpaka::memcpy(queue, x_d, x_h);
    alpaka::memcpy(queue, y_d, y_h);

    alpaka::wait(queue);

    const int numBlocks = 4;
    const int numThreadsPerBlock = len / numBlocks ;

    const auto workDiv = make_workdiv<Acc1D>(numBlocks, numThreadsPerBlock);

    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv, scaleKernel{}, x_d.data(), z_d.data()));

    alpaka::memcpy(queue, z_h, z_d);

    alpaka::wait(queue);

    int fails = 0;
    constexpr data_t epsilon = 1e-12; 

    for( int i = 0; i < len*nSrc; i++) {
      const data_t res = s * x_view[i];
      //
      const data_t r = res - z_view[i];
      //
      if (std::abs(r) > epsilon) fails++;
    }   
  
    assert(fails == 0);

    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv, addKernel(), x_d.data(), y_d.data(), z_d.data()));

    alpaka::memcpy(queue, z_h, z_d);

    alpaka::wait(queue);

    for( int i = 0; i < len*nSrc; i++) {
      const data_t res = x_view[i] + y_view[i];
      //
      const data_t r = res - z_view[i];
      //
      if (std::abs(r) > epsilon) fails++;
    }  

    assert(fails == 0);

    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv, subKernel(), x_d.data(), y_d.data(), z_d.data()));

    alpaka::memcpy(queue, z_h, z_d);

    alpaka::wait(queue);

    for( int i = 0; i < len*nSrc; i++) {
      const data_t res = x_view[i] - y_view[i];
      //
      const data_t r = res - z_view[i];
      //
      if (std::abs(r) > epsilon) fails++;
    }

    assert(fails == 0);

    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv, axpyKernel(), x_d.data(), y_d.data(), z_d.data()));

    alpaka::memcpy(queue, z_h, z_d);

    alpaka::wait(queue);

    for( int i = 0; i < len*nSrc; i++) {
      const data_t res = s * x_view[i] + y_view[i];
      //
      const data_t r = res - z_view[i];
      //
      if (std::abs(r) > epsilon) fails++;
    }

    assert(fails == 0);

    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv, dotKernel(), x_d.data(), y_d.data(), result_d.data()));

    alpaka::memcpy(queue, result_h, result_d);

    alpaka::wait(queue);

    for( int i = 0; i < len; i++) {
      data_t res = x_view[i] * y_view[i];
      for (int j = 1; j < nSrc; j++) res += x_view[i+j*len] * y_view[i+j*len];
      //
      const data_t r = res - result_view[i];
      //
      if (std::abs(r) > epsilon) fails++;
    }

    assert(fails == 0);    
  }
  std::cout << "TEST PASSED" << std::endl;
  return 0;
}
