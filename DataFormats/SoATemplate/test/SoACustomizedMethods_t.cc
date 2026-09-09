#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "SoADefinition_CustomizedMethods.h"

TEST_CASE("SoACustomizedMethods") {
  // common number of elements for the SoAs
  const std::size_t elems = 10;

  // buffer size
  const std::size_t bufferSize = SoA::computeDataSize(elems);

  // memory buffer for the SoA
  std::unique_ptr<std::byte, decltype(std::free) *> buffer{
      reinterpret_cast<std::byte *>(aligned_alloc(SoA::alignment, bufferSize)), std::free};

  // SoA objects
  SoA soa{buffer.get(), elems};
  SoAView view{soa};
  SoAConstView const_view{soa};

  // fill up
  for (size_t i = 0; i < elems; i++) {
    view[i].x() = static_cast<float>(i);
    view[i].y() = static_cast<float>(i) * 2.0f;
    view[i].z() = static_cast<float>(i) * 3.0f;
    view[i].v_x() = static_cast<double>(i);
    view[i].v_y() = static_cast<double>(i) * 20;
    view[i].v_z() = static_cast<double>(i) * 30;
  }
  view.detectorType() = 42;

  SECTION("ConstElement methods") {
    // arrays of norms
    std::array<float, elems> position_norms;
    std::array<double, elems> velocity_norms;

    // Check for the correctness of the square_norm() functions
    for (size_t i = 0; i < elems; i++) {
      position_norms[i] = sqrt(const_view[i].x() * const_view[i].x() + const_view[i].y() * const_view[i].y() +
                               const_view[i].z() * const_view[i].z());
      velocity_norms[i] = sqrt(const_view[i].v_x() * const_view[i].v_x() + const_view[i].v_y() * const_view[i].v_y() +
                               const_view[i].v_z() * const_view[i].v_z());
      REQUIRE(position_norms[i] == const_view[i].square_norm_position());
      REQUIRE(velocity_norms[i] == const_view[i].square_norm_velocity());
    }
  }

  SECTION("Element methods") {
    // array of times
    std::array<double, elems> times;

    // Check for the correctness of the time() function
    times[0] = 0.;
    for (size_t i = 0; i < elems; i++) {
      if (not(i == 0))
        times[i] = view[i].x() / view[i].v_x();
      REQUIRE(times[i] == SoAView::const_element::time(view[i].x(), view[i].v_x()));
    }

    // normalise the particles data
    for (size_t i = 0; i < elems; i++) {
      view[i].normalise();
    }

    // Check for the norm equal to 1 except for the first element
    REQUIRE(view[0].square_norm_position() == 0.f);
    REQUIRE(view[0].square_norm_velocity() == 0.);
    for (size_t i = 1; i < elems; i++) {
      REQUIRE_THAT(view[i].square_norm_position(), Catch::Matchers::WithinAbs(1.f, 1.e-6));
      REQUIRE_THAT(view[i].square_norm_velocity(), Catch::Matchers::WithinAbs(1., 1.e-9));
    }
  }

  const auto points_sizes = std::array<cms::soa::size_type, 2>{{2, 2}};
  const std::size_t blocksBufferSize = Points::computeDataSize(points_sizes);

  std::unique_ptr<std::byte, decltype(std::free) *> points_buffer{
      reinterpret_cast<std::byte *>(aligned_alloc(Points::alignment, blocksBufferSize)), std::free};

  Points points(points_buffer.get(), points_sizes);
  PointsView points_view{points};
  PointsConstView points_const_view{points};
  points_view.position()[0].x() = 2.f;
  points_view.position()[0].y() = 4.f;
  points_view.position()[0].z() = 3.f;
  points_view.position()[1].x() = 1.f;
  points_view.position()[1].y() = 1.f;
  points_view.position()[1].z() = 1.f;

  SECTION("View methods") { REQUIRE(points_const_view.distance2(0, 1) == 14.f); }

  SECTION("ConstView methods") {
    points_view.velocity()[0].vx() = 1.f;
    points_view.velocity()[0].vy() = 3.f;
    points_view.velocity()[0].vz() = 5.f;
    const auto time = .5f;
    points_view.update_position(0, time);
    REQUIRE(points_view.position()[0].x() == 2.5f);
    REQUIRE(points_view.position()[0].y() == 5.5f);
    REQUIRE(points_view.position()[0].z() == 5.5f);
  }
}
