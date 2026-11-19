#include <Eigen/Core>

#include <cmath>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/CaloRecHit/test/CaloClusterSoATestValues.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"

using ALPAKA_ACCELERATOR_NAMESPACE::Platform;
using ALPAKA_ACCELERATOR_NAMESPACE::Queue;

TEST_CASE("CaloCluster SoA collection: sizes and copy to the device and back", "[CaloClusterSoA]") {
  const auto& devices = cms::alpakatools::devices<Platform>();
  REQUIRE(not devices.empty());
  Queue queue(devices.front());

  const int32_t n = GENERATE(0, 1, 2, 5);
  reco::CaloClusterHostCollection host(queue, n, n, n, n);
  caloclustertest::fill(host.view());
  REQUIRE_NOTHROW(caloclustertest::check(host.const_view(), n));

  ALPAKA_ACCELERATOR_NAMESPACE::reco::CaloClusterDeviceCollection device(queue, n, n, n, n);
  alpaka::memcpy(queue, device.buffer(), host.const_buffer());
  reco::CaloClusterHostCollection back(queue, n, n, n, n);
  alpaka::memcpy(queue, back.buffer(), device.const_buffer());
  alpaka::wait(queue);
  REQUIRE_NOTHROW(caloclustertest::check(back.const_view(), n));
}

TEST_CASE("CaloCluster SoA collection: r, phi and eta", "[CaloClusterSoA]") {
  reco::CaloClusterHostCollection host(3, 3, 3, 3);
  auto view = host.view();
  const float positions[3][3] = {{3.f, 4.f, 0.f}, {0.f, 2.f, 2.f}, {0.f, 0.f, -5.f}};
  for (int i = 0; i < 3; ++i) {
    view.position()[i].x() = positions[i][0];
    view.position()[i].y() = positions[i][1];
    view.position()[i].z() = positions[i][2];
  }
  const auto constView = host.const_view();
  CHECK(constView.r(0) == Catch::Approx(5.f));
  CHECK(constView.eta(0) == Catch::Approx(0.f));
  CHECK(constView.phi(1) == Catch::Approx(M_PI_2));
  CHECK(constView.eta(1) == Catch::Approx(std::asinh(1.f)));
  // On the z axis the cluster has r = 0 and an infinite eta with the sign of z.
  CHECK(constView.r(2) == 0.f);
  CHECK(std::isinf(constView.eta(2)));
  CHECK(constView.eta(2) < 0.f);
}
