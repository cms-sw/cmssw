#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(SoATemplate,
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z),
                    SOA_COLUMN(double, v_x),
                    SOA_COLUMN(double, v_y),
                    SOA_COLUMN(double, v_z),

                    SOA_ELEMENT_METHODS(

                        void normalise() {
                          float norm_position = square_norm_position();
                          if (norm_position > 0.0f) {
                            x() /= norm_position;
                            y() /= norm_position;
                            z() /= norm_position;
                          };
                          double norm_velocity = square_norm_velocity();
                          if (norm_velocity > 0.0f) {
                            v_x() /= norm_velocity;
                            v_y() /= norm_velocity;
                            v_z() /= norm_velocity;
                          };
                        }),

                    SOA_CONST_ELEMENT_METHODS(
                        float square_norm_position() const { return sqrt(x() * x() + y() * y() + z() * z()); };

                        double square_norm_velocity()
                            const { return sqrt(v_x() * v_x() + v_y() * v_y() + v_z() * v_z()); };

                        template <typename T1, typename T2>
                        static auto time(T1 pos, T2 vel) {
                          if (not(vel == 0))
                            return pos / vel;
                          return 0.;
                        }),

                    SOA_SCALAR(int, detectorType))

using SoA = SoATemplate<>;
using SoAView = SoA::View;
using SoAConstView = SoA::ConstView;

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

  SECTION("ConstView methods") {
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

  SECTION("View methods") {
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
}
