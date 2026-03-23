#include <iostream>
#include <sstream>

#include <Eigen/Core>
#include <Eigen/Dense>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(SoATemplate,
                    // columns: one value per element
                    SOA_COLUMN(double, x),
                    SOA_COLUMN(double, y),
                    SOA_COLUMN(double, z),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, a),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, b),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, r),
                    SOA_COLUMN(uint16_t, colour),
                    SOA_COLUMN(int32_t, value),
                    SOA_COLUMN(double *, py),
                    SOA_COLUMN(uint32_t, count),
                    SOA_COLUMN(uint32_t, anotherCount),

                    // scalars: one value for the whole structure
                    SOA_SCALAR(const char *, description),
                    SOA_SCALAR(uint32_t, someNumber));

using SoA = SoATemplate<64, true>;

TEST_CASE("Stream SoA") {
  const std::size_t slSize = 32;
  const std::size_t slBufferSize = SoA::computeDataSize(slSize);
  // memory buffer aligned according to the layout requirements
  std::unique_ptr<std::byte, decltype(std::free) *> slBuffer{
      reinterpret_cast<std::byte *>(aligned_alloc(SoA::alignment, slBufferSize)), std::free};
  // SoA layout
  SoA soa{slBuffer.get(), slSize};

  std::ostringstream expected;
  expected << "SoATemplate(32 elements, byte alignement= 64, @" << static_cast<void *>(slBuffer.get()) << "): \n"
           << "  sizeof(SoATemplate): 176\n"
           << " Column x at offset 0 has size 256 and padding 0\n"
           << " Column y at offset 256 has size 256 and padding 0\n"
           << " Column z at offset 512 has size 256 and padding 0\n"
           << " Eigen value a at offset 768 has dimension (3 x 1) and per column size 256 and padding 0\n"
           << " Eigen value b at offset 1536 has dimension (3 x 1) and per column size 256 and padding 0\n"
           << " Eigen value r at offset 2304 has dimension (3 x 1) and per column size 256 and padding 0\n"
           << " Column colour at offset 3072 has size 64 and padding 0\n"
           << " Column value at offset 3136 has size 128 and padding 0\n"
           << " Column py at offset 3264 has size 256 and padding 0\n"
           << " Column count at offset 3520 has size 128 and padding 0\n"
           << " Column anotherCount at offset 3648 has size 128 and padding 0\n"
           << " Scalar description at offset 3776 has size 8 and padding 56\n"
           << " Scalar someNumber at offset 3840 has size 4 and padding 60\n"
           << "Final offset = 3904 computeDataSize(...): 3904\n\n";

  std::ostringstream oss;
  oss << soa;

  REQUIRE(oss.str() == expected.str());
}
