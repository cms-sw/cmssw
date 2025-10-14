#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(SoATemplate,
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z),

                    SOA_CONST_METHODS(

                        SOA_HOST_ONLY std::array<float, 3> centroid() const {
                          float x_sum = 0.f, y_sum = 0.f, z_sum = 0.f;
                          auto n = this->metadata().size();

                          for (int i = 0; i < n; ++i) {
                            x_sum += x(i);
                            y_sum += y(i);
                            z_sum += z(i);
                          }
                          return std::array<float, 3>{{x_sum / n, y_sum / n, z_sum / n}};
                        }),

                    SOA_CONST_ELEMENT_METHODS(

                        float norm_position() const { return x() * x() + y() * y() + z() * z(); }),

                    SOA_METHODS(

                        SOA_HOST_ONLY void sortByDistance() {
                          auto n = this->metadata().size();

                          // build permutation based on norm
                          std::vector<int> indices(n);
                          std::iota(indices.begin(), indices.end(), 0);

                          std::sort(indices.begin(), indices.end(), [&](int i, int j) {
                            return (*this)[i].norm_position() < (*this)[j].norm_position();
                          });

                          // apply permutation to x,y,z
                          std::vector<float> new_x(n), new_y(n), new_z(n);
                          for (int k = 0; k < n; ++k) {
                            new_x[k] = x(indices[k]);
                            new_y[k] = y(indices[k]);
                            new_z[k] = z(indices[k]);
                          }

                          for (int k = 0; k < n; ++k) {
                            x(k) = new_x[k];
                            y(k) = new_y[k];
                            z(k) = new_z[k];
                          }
                          sorted() = true;
                        }

                        SOA_HOST_DEVICE void scale(const float factor) {
                          auto n = this->metadata().size();
                          for (int i = 0; i < n; ++i) {
                            x(i) *= factor;
                            y(i) *= factor;
                            z(i) *= factor;
                          }
                        }),

                    SOA_SCALAR(bool, sorted))

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

  // random initialization
  std::mt19937 rand(123456u);
  std::uniform_real_distribution<float> dist(-50.f, 50.f);

  for (std::size_t i = 0; i < elems; i++) {
    view[i].x() = dist(rand);
    view[i].y() = dist(rand);
    view[i].z() = dist(rand);
  }
  view.sorted() = false;

  SECTION("__host__ methods") {
    // compute centroid before sorting
    auto centroid = const_view.centroid();

    // sort the SoA
    REQUIRE_FALSE(const_view.sorted());
    view.sortByDistance();
    REQUIRE(const_view.sorted());

    // check that sorting worked as expected
    float prev = -std::numeric_limits<float>::infinity();
    for (std::size_t k = 0; k < elems; k++) {
      float cur = const_view[k].norm_position();
      REQUIRE(cur >= prev);
      prev = cur;
    }

    // centroid preserved
    auto new_centroid = const_view.centroid();
    REQUIRE(new_centroid[0] == Approx(centroid[0]));
    REQUIRE(new_centroid[1] == Approx(centroid[1]));
    REQUIRE(new_centroid[2] == Approx(centroid[2]));
  }

  SECTION("__host__ __device__ method") {
    // make a deep copy of the SoA into a new one
    std::unique_ptr<std::byte, decltype(std::free) *> buffer2{
        reinterpret_cast<std::byte *>(aligned_alloc(SoA::alignment, bufferSize)), std::free};

    SoA soa2{buffer2.get(), elems};
    SoAView original_view{soa2};
    soa2.deepCopy(view);

    // scale the original SoA
    const float factor = 2.5f;
    view.scale(factor);

    // check that scaling worked as expected
    for (std::size_t i = 0; i < elems; i++) {
      REQUIRE(view[i].x() == Approx(original_view[i].x() * factor));
      REQUIRE(view[i].y() == Approx(original_view[i].y() * factor));
      REQUIRE(view[i].z() == Approx(original_view[i].z() * factor));
    }
  }
}