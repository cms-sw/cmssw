#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <hip/hip_runtime.h>

#include "HeterogeneousCore/ROCmUtilities/interface/hipCheck.h"
#include "HeterogeneousCore/ROCmUtilities/interface/requireDevices.h"

#include "SoADefinition_CustomizedMethods.h"

__global__ void calculateNorm(SoAConstView soaConstView, float* resultNorm, double* resultVelNorm) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= soaConstView.metadata().size())
    return;

  resultNorm[i] = soaConstView[i].square_norm_position();
  resultVelNorm[i] = soaConstView[i].square_norm_velocity();
}

__global__ void checkNormalise(SoAView soaView, double* checkTimesFunction) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= soaView.metadata().size())
    return;

  checkTimesFunction[i] = SoAView::const_element::time(soaView[i].x(), soaView[i].v_x());
  soaView[i].normalise();
}

__global__ void checkPointsDistance(PointsConstView view, bool* result) { *result &= (view.distance2(0, 1) == 14.f); }

__global__ void checkPointsPositionUpdate(PointsView view, float time, bool* result) {
  view.update_position(0, time);
  *result &= (view.position()[0].x() == 2.5f);
  *result &= (view.position()[0].y() == 5.5f);
  *result &= (view.position()[0].z() == 5.5f);
}

TEST_CASE("SoACustomizedMethods hip", "[SoACustomizedMethods][hip]") {
  // common number of elements for the SoAs
  const std::size_t elems = 10;

  // buffer size
  const std::size_t bufferSize = SoA::computeDataSize(elems);

  std::byte* h_buf = nullptr;
  hipCheck(hipHostMalloc(&h_buf, bufferSize));
  SoA h_soahdLayout(h_buf, elems);
  SoAView h_view(h_soahdLayout);
  SoAConstView h_Constview(h_soahdLayout);

  // fill up
  for (size_t i = 0; i < elems; i++) {
    h_view[i].x() = static_cast<float>(i);
    h_view[i].y() = static_cast<float>(i) * 2.0f;
    h_view[i].z() = static_cast<float>(i) * 3.0f;
    h_view[i].v_x() = static_cast<double>(i);
    h_view[i].v_y() = static_cast<double>(i) * 20;
    h_view[i].v_z() = static_cast<double>(i) * 30;
  }
  h_view.detectorType() = 42;

  std::byte* d_buf = nullptr;
  hipCheck(hipMalloc(&d_buf, bufferSize));
  SoA d_soahdLayout(d_buf, elems);
  SoAView d_view(d_soahdLayout);
  SoAConstView d_Constview(d_soahdLayout);

  std::vector<float> h_position_norms(elems);
  std::vector<double> h_velocity_norms(elems);
  std::vector<double> h_times(elems);

  float* d_position_norms;
  double* d_velocity_norms;
  double* d_times;

  hipCheck(hipMalloc(&d_position_norms, elems * sizeof(float)));
  hipCheck(hipMalloc(&d_velocity_norms, elems * sizeof(double)));
  hipCheck(hipMalloc(&d_times, elems * sizeof(double)));

  // Host → Device copy
  hipCheck(hipMemcpy(d_buf, h_buf, bufferSize, hipMemcpyHostToDevice));

  SECTION("ConstElement methods HIP") {
    calculateNorm<<<(elems + 255) / 256, 256>>>(d_Constview, d_position_norms, d_velocity_norms);

    hipCheck(hipMemcpy(h_position_norms.data(), d_position_norms, elems * sizeof(float), hipMemcpyDeviceToHost));
    hipCheck(hipMemcpy(h_velocity_norms.data(), d_velocity_norms, elems * sizeof(double), hipMemcpyDeviceToHost));

    // Check for the correctness of the square_norm() functions
    for (size_t i = 0; i < elems; i++) {
      const float position_norm =
          sqrt(h_Constview[i].x() * h_Constview[i].x() + h_Constview[i].y() * h_Constview[i].y() +
               h_Constview[i].z() * h_Constview[i].z());
      const double velocity_norm =
          sqrt(h_Constview[i].v_x() * h_Constview[i].v_x() + h_Constview[i].v_y() * h_Constview[i].v_y() +
               h_Constview[i].v_z() * h_Constview[i].v_z());
      REQUIRE(h_position_norms[i] == position_norm);
      REQUIRE(h_velocity_norms[i] == velocity_norm);
    }
  }

  SECTION("Element methods HIP") {
    std::array<double, elems> times;

    // Check for the correctness of the time() function
    times[0] = 0.;
    for (size_t i = 0; i < elems; i++) {
      if (!(i == 0))
        times[i] = h_view[i].x() / h_view[i].v_x();
    }

    checkNormalise<<<(elems + 255) / 256, 256>>>(d_view, d_times);

    hipCheck(hipMemcpy(h_times.data(), d_times, elems * sizeof(double), hipMemcpyDeviceToHost));
    hipCheck(hipMemcpy(h_buf, d_buf, bufferSize, hipMemcpyDeviceToHost));

    // Check for the correctness of the time() function
    for (size_t i = 0; i < elems; i++) {
      REQUIRE(h_times[i] == times[i]);
    }

    REQUIRE(h_view[0].square_norm_position() == 0.f);
    REQUIRE(h_view[0].square_norm_velocity() == 0.);
    for (size_t i = 1; i < elems; i++) {
      REQUIRE_THAT(h_view[i].square_norm_position(), Catch::Matchers::WithinAbs(1.f, 1.e-6));
      REQUIRE_THAT(h_view[i].square_norm_velocity(), Catch::Matchers::WithinAbs(1., 1.e-9));
    }
  }

  const auto points_sizes = std::array<cms::soa::size_type, 2>{{2, 2}};
  const std::size_t blocksBufferSize = Points::computeDataSize(points_sizes);

  std::byte* points_buffer_host = nullptr;
  hipCheck(hipHostMalloc(&points_buffer_host, blocksBufferSize));

  Points h_points(points_buffer_host, points_sizes);
  PointsView h_points_view{h_points};
  h_points_view.position()[0].x() = 2.f;
  h_points_view.position()[0].y() = 4.f;
  h_points_view.position()[0].z() = 3.f;
  h_points_view.position()[1].x() = 1.f;
  h_points_view.position()[1].y() = 1.f;
  h_points_view.position()[1].z() = 1.f;
  h_points_view.velocity()[0].vx() = 1.f;
  h_points_view.velocity()[0].vy() = 3.f;
  h_points_view.velocity()[0].vz() = 5.f;

  std::byte* points_buffer_device = nullptr;
  hipCheck(hipMalloc(&points_buffer_device, blocksBufferSize));

  Points d_points(points_buffer_device, points_sizes);
  PointsView d_points_view{d_points};
  PointsConstView d_points_const_view{d_points};

  hipCheck(hipMemcpy(points_buffer_device, points_buffer_host, blocksBufferSize, hipMemcpyHostToDevice));

  bool* d_result = nullptr;
  hipCheck(hipMalloc(&d_result, sizeof(bool)));
  hipCheck(hipMemset(d_result, 1, sizeof(bool)));

  SECTION("View methods") { checkPointsDistance<<<1, 1>>>(d_points_const_view, d_result); }

  SECTION("ConstView methods") {
    const auto time = .5f;
    checkPointsPositionUpdate<<<1, 1>>>(d_points_view, time, d_result);
  }

  bool h_result;
  hipCheck(hipMemcpy(&h_result, d_result, sizeof(bool), hipMemcpyDeviceToHost));
  REQUIRE(h_result);

  // ===== cleanup =====
  hipCheck(hipFree(d_position_norms));
  hipCheck(hipFree(d_velocity_norms));
  hipCheck(hipFree(d_times));
  hipCheck(hipFree(d_buf));
  hipCheck(hipFree(points_buffer_device));
  hipCheck(hipFree(d_result));
  hipCheck(hipFreeHost(h_buf));
  hipCheck(hipFreeHost(points_buffer_host));
}
