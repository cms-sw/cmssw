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

using Vec3d = cms::alpakatools::math::Vector<double, 3>;
using data_t = typename Vec3d::value_type;

constexpr int len{1 << 12};
constexpr int nSrc{Vec3d::N};
constexpr data_t c{3.14};

struct axKernel {
  const data_t a;

  constexpr axKernel() : a(c) {}

  ALPAKA_FN_ACC void operator()(Acc1D const& acc, const data_t* x, data_t* z) const {
    for (auto i : uniform_elements(acc)) {
      const Vec3d xvec(x[i], x[i + len], x[i + len * 2]);

      const Vec3d zvec = cms::alpakatools::math::ax(a, xvec);

      CMS_UNROLL_LOOP
      for (int n = 0; n < nSrc; n++) {
        z[i + n * len] = zvec[n];
      }
    }
  }
};

struct xpyKernel {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, const data_t* x, const data_t* y, data_t* z) const {
    for (auto i : uniform_elements(acc)) {
      const Vec3d xvec(x[i], x[i + len], x[i + len * 2]);
      const Vec3d yvec(y[i], y[i + len], y[i + len * 2]);

      const Vec3d zvec = cms::alpakatools::math::xpy(xvec, yvec);

      CMS_UNROLL_LOOP
      for (int n = 0; n < nSrc; n++) {
        z[i + n * len] = zvec[n];
      }
    }
  }
};

struct xmyKernel {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, const data_t* x, const data_t* y, data_t* z) const {
    for (auto i : uniform_elements(acc)) {
      const Vec3d xvec(x[i], x[i + len], x[i + len * 2]);
      const Vec3d yvec(y[i], y[i + len], y[i + len * 2]);

      const Vec3d zvec = cms::alpakatools::math::xmy(xvec, yvec);

      CMS_UNROLL_LOOP
      for (int n = 0; n < nSrc; n++) {
        z[i + n * len] = zvec[n];
      }
    }
  }
};

struct axpyKernel {
  const data_t a;

  constexpr axpyKernel() : a(c) {}

  ALPAKA_FN_ACC void operator()(Acc1D const& acc, const data_t* x, const data_t* y, data_t* z) const {
    for (auto i : uniform_elements(acc)) {
      const Vec3d xvec(x[i], x[i + len], x[i + len * 2]);
      const Vec3d yvec(y[i], y[i + len], y[i + len * 2]);

      const Vec3d zvec = cms::alpakatools::math::axpy(a, xvec, yvec);

      CMS_UNROLL_LOOP
      for (int n = 0; n < nSrc; n++) {
        z[i + n * len] = zvec[n];
      }
    }
  }
};

struct normalizeKernel {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, data_t* x) const {
    for (auto i : uniform_elements(acc)) {
      Vec3d xvec(x[i], x[i + len], x[i + len * 2]);

      xvec.normalize(acc);

      CMS_UNROLL_LOOP
      for (int n = 0; n < nSrc; n++) {
        x[i + n * len] = xvec[n];
      }
    }
  }
};

struct normKernel {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, const data_t* x, data_t* r) const {
    for (auto i : uniform_elements(acc)) {
      const Vec3d xvec(x[i], x[i + len], x[i + len * 2]);

      r[i] = xvec.norm(acc);
    }
  }
};

struct partialNormKernel {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, const data_t* x, data_t* r) const {
    for (auto i : uniform_elements(acc)) {
      const Vec3d xvec(x[i], x[i + len], x[i + len * 2]);

      r[i] = xvec.template partial_norm<Acc1D, 2>(acc);
    }
  }
};

struct dotKernel {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, const data_t* x, const data_t* y, data_t* r) const {
    for (auto i : uniform_elements(acc)) {
      const Vec3d xvec(x[i], x[i + len], x[i + len * 2]);
      const Vec3d yvec(y[i], y[i + len], y[i + len * 2]);

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

  using data_t = double;

  constexpr int N = 1 << 12;
  constexpr int nSrc = 3;

  // run the test on each device
  for (auto const& device : devices) {
    auto queue = Queue(device);

    auto x_h = make_host_buffer<data_t[]>(queue, N * nSrc);
    auto x_d = make_device_buffer<data_t[]>(queue, N * nSrc);

    auto y_h = make_host_buffer<data_t[]>(queue, N * nSrc);
    auto y_d = make_device_buffer<data_t[]>(queue, N * nSrc);

    auto z_h = make_host_buffer<data_t[]>(queue, N * nSrc);
    auto z_d = make_device_buffer<data_t[]>(queue, N * nSrc);

    auto result_d = make_device_buffer<data_t[]>(queue, N);

    auto _x = x_h.data();
    auto _y = y_h.data();
    auto _z = z_h.data();

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<data_t> distr(0.0, 1.0);

    for (int i = 0; i < N * nSrc; i++) {
      _x[i] = distr(gen);
      _y[i] = distr(gen);
    }

    // copy the object to the device.
    alpaka::memcpy(queue, x_d, x_h);
    alpaka::memcpy(queue, y_d, y_h);
    alpaka::wait(queue);

    const int numBlocks = 4;
    const int numThreadsPerBlock = len / numBlocks;

    const auto workDiv = make_workdiv<Acc1D>(numBlocks, numThreadsPerBlock);

    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv, axKernel{}, x_d.data(), z_d.data()));
    alpaka::wait(queue);

    alpaka::memcpy(queue, z_h, z_d);

    int fails = 0;
    constexpr data_t epsilon = 1e-16;

    for (int i = 0; i < len * nSrc; i++) {
      _x[i] = c * _x[i];
      //
      const data_t r = _x[i] - _z[i];
      //
      if (std::abs(r) > epsilon)
        fails++;
    }

    assert(fails == 0);

    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv, xpyKernel(), x_d.data(), y_d.data(), z_d.data()));
    alpaka::wait(queue);

    alpaka::memcpy(queue, z_h, z_d);

    for (int i = 0; i < len * nSrc; i++) {
      const data_t res = _x[i] + _y[i];
      //
      const data_t r = res - _z[i];
      //
      if (std::abs(r) > epsilon)
        fails++;
    }

    assert(fails == 0);

    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv, xmyKernel(), x_d.data(), y_d.data(), z_d.data()));
    alpaka::wait(queue);

    alpaka::memcpy(queue, z_h, z_d);

    for (int i = 0; i < len * nSrc; i++) {
      const data_t res = _x[i] - _y[i];
      //
      const data_t r = res - _z[i];
      //
      if (std::abs(r) > epsilon)
        fails++;
    }

    assert(fails == 0);

    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv, axpyKernel(), x_d.data(), y_d.data(), z_d.data()));
    alpaka::wait(queue);

    alpaka::memcpy(queue, z_h, z_d);

    for (int i = 0; i < len * nSrc; i++) {
      const data_t res = c * _x[i] + _y[i];
      //
      const data_t r = res - _z[i];
      //
      if (std::abs(r) > epsilon)
        fails++;
    }

    assert(fails == 0);
  }
  std::cout << "TEST PASSED" << std::endl;
  return 0;
}
