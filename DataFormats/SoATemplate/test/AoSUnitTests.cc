#include <Eigen/Core>
#include <Eigen/Dense>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include <iostream>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(SoATemplate,
                    SOA_SCALAR(int8_t, s1),
                    SOA_COLUMN(float, xPos),
                    SOA_COLUMN(float, yPos),
                    SOA_COLUMN(float, zPos),
                    SOA_SCALAR(float, s2),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, candidateDirection),
                    SOA_COLUMN(double *, py),
                    SOA_SCALAR(int64_t, s3),
                    SOA_SCALAR(double, s4),
                    SOA_SCALAR(const char *, s5))

using SoA = SoATemplate<>;
using SoAView = SoA::View;
using SoAConstView = SoA::ConstView;

GENERATE_SOA_LAYOUT(SoATemplateOnlyScalars,
                    SOA_SCALAR(int8_t, s1),
                    SOA_SCALAR(float, s2),
                    SOA_SCALAR(int64_t, s3),
                    SOA_SCALAR(double, s4))

using SoAOnlyScalars = SoATemplateOnlyScalars<>;
using SoAViewOnlyScalar = SoAOnlyScalars::View;
using SoAConstViewOnlyScalar = SoAOnlyScalars::ConstView;

TEST_CASE("SoAToAoS") {
  // common number of elements for the SoAs
  const std::size_t elems = 16;

  SECTION("Base test with SoATemplate") {
    // buffer sizes
    const std::size_t soaBufferSize = SoA::computeDataSize(elems);
    const std::size_t aosBufferSize = SoA::AoSWrapper::computeDataSize(elems);
    // The AoS is an array of SoA::Metadata::value_element
    // The struct SoA::Metadata::value_element is 3*sizeof(float) + sizeof(Eigen::Vector3d) + sizeof(double*) = 44 bytes. This is padded to 48 bytes on
    // So the total memory is 48 bytes * elems + the size of the scalar members
    const std::size_t expectedBufferSize = sizeof(SoA::Metadata::value_element) * elems + sizeof(int8_t) +
                                           sizeof(float) + sizeof(int64_t) + sizeof(double) + sizeof(const char *);
    REQUIRE(expectedBufferSize == aosBufferSize);

    // memory buffer for the SoA
    std::unique_ptr<std::byte, decltype(std::free) *> soaBuffer{
        reinterpret_cast<std::byte *>(aligned_alloc(SoA::alignment, soaBufferSize)), std::free};

    std::unique_ptr<std::byte, decltype(std::free) *> aosBuffer{
        reinterpret_cast<std::byte *>(std::malloc(aosBufferSize)), std::free};

    // SoA Layout
    SoA soa{soaBuffer.get(), elems};

    // SoA Views
    SoAView soaView{soa};
    SoAConstView soaConstView{soa};

    std::vector<double> pydata(elems * 3, 0.0);
    for (size_t i = 0; i < pydata.size(); i++) {
      pydata[i] = static_cast<double>(i) + 0.01;
    }

    // fill up
    for (size_t i = 0; i < elems; i++) {
      soaView[i].xPos() = static_cast<float>(i);
      soaView[i].yPos() = static_cast<float>(i) + 0.1f;
      soaView[i].zPos() = static_cast<float>(i) + 0.2f;
      soaView[i].candidateDirection()(0) = static_cast<double>(i) + 0.3;
      soaView[i].candidateDirection()(1) = static_cast<double>(i) + 0.4;
      soaView[i].candidateDirection()(2) = static_cast<double>(i) + 0.5;
      soaView[i].py() = &pydata[3 * i];
    }
    soaView.s1() = 100;
    soaView.s2() = 42.42f;
    soaView.s3() = (int64_t(1) << 42) + 852516352;
    soaView.s4() = static_cast<double>((int64_t(1) << 42) + 8.52516352);
    soaView.s5() = "Testing";

    // Copy to AoS
    SoA::AoSWrapper aos{aosBuffer.get(), elems};
    SoA::AoSWrapper::View aosView{aos};
    SoA::AoSWrapper::ConstView aosConstView{aos};

    for (size_t i = 0; i < elems; i++)
      aosView.transpose(soaView, i);

    // Check that the sizes are the same
    REQUIRE(soaConstView.metadata().size() == aosConstView.metadata().size());
    REQUIRE(elems == aosConstView.metadata().size());

    for (size_t i = 0; i < elems; i++) {
      REQUIRE_THAT(aosConstView.xPos()[i], Catch::Matchers::WithinAbs(static_cast<float>(i), 1.e-6));
      REQUIRE_THAT(aosConstView.yPos()[i], Catch::Matchers::WithinAbs(static_cast<float>(i) + 0.1f, 1.e-6));
      REQUIRE_THAT(aosConstView.zPos()[i], Catch::Matchers::WithinAbs(static_cast<float>(i) + 0.2f, 1.e-6));
      REQUIRE_THAT(aosConstView[i].candidateDirection()(0),
                   Catch::Matchers::WithinAbs(static_cast<double>(i) + 0.3, 1.e-6));
      REQUIRE_THAT(aosConstView[i].candidateDirection()(1),
                   Catch::Matchers::WithinAbs(static_cast<double>(i) + 0.4, 1.e-6));
      REQUIRE_THAT(aosConstView[i].candidateDirection()(2),
                   Catch::Matchers::WithinAbs(static_cast<double>(i) + 0.5, 1.e-6));

      const double *py = aosConstView[i].py();
      for (size_t j = 0; j < 3; j++) {
        REQUIRE_THAT(py[j], Catch::Matchers::WithinAbs(static_cast<double>(3 * i + j) + 0.01, 1.e-6));
      }
    }
    REQUIRE(aosConstView.s1() == 100);
    REQUIRE_THAT(aosConstView.s2(), Catch::Matchers::WithinAbs(42.42f, 1.e-6));
    REQUIRE(aosConstView.s3() == (int64_t(1) << 42) + 852516352);
    REQUIRE_THAT(aosConstView.s4(),
                 Catch::Matchers::WithinAbs(static_cast<double>((int64_t(1) << 42) + 8.52516352), 1.e-6));
    REQUIRE(std::string(aosConstView.s5()) == "Testing");

    const int underflow = -1;
    const int overflow = aosConstView.metadata().size();
    // Check for under-and overflow in the row accessor
    REQUIRE_THROWS_AS(aosConstView[underflow], std::out_of_range);
    REQUIRE_THROWS_AS(aosConstView[overflow], std::out_of_range);
    // Check for under-and overflow in the row accessor
    REQUIRE_THROWS_AS(aosView[underflow], std::out_of_range);
    REQUIRE_THROWS_AS(aosView[overflow], std::out_of_range);

    // Check that the AoS memory layout is as expected
    for (size_t i = 0; i < elems; i++) {
      float xPos;
      float yPos;
      float zPos;

      double candidateDirection0;
      double candidateDirection1;
      double candidateDirection2;

      double *py;

      std::memcpy(&xPos, aosBuffer.get() + 0 + i * 48, sizeof(float));
      std::memcpy(&yPos, aosBuffer.get() + 4 + i * 48, sizeof(float));
      std::memcpy(&zPos, aosBuffer.get() + 8 + i * 48, sizeof(float));
      // The jump of 8 bytes is due to padding
      std::memcpy(&candidateDirection0, aosBuffer.get() + 16 + i * 48, sizeof(double));
      std::memcpy(&candidateDirection1, aosBuffer.get() + 24 + i * 48, sizeof(double));
      std::memcpy(&candidateDirection2, aosBuffer.get() + 32 + i * 48, sizeof(double));

      std::memcpy(&py, aosBuffer.get() + 40 + i * 48, sizeof(double *));

      REQUIRE_THAT(xPos, Catch::Matchers::WithinAbs(static_cast<float>(i), 1.e-6));
      REQUIRE_THAT(yPos, Catch::Matchers::WithinAbs(static_cast<float>(i) + 0.1f, 1.e-6));
      REQUIRE_THAT(zPos, Catch::Matchers::WithinAbs(static_cast<float>(i) + 0.2f, 1.e-6));
      REQUIRE_THAT(candidateDirection0, Catch::Matchers::WithinAbs(static_cast<double>(i) + 0.3, 1.e-6));
      REQUIRE_THAT(candidateDirection1, Catch::Matchers::WithinAbs(static_cast<double>(i) + 0.4, 1.e-6));
      REQUIRE_THAT(candidateDirection2, Catch::Matchers::WithinAbs(static_cast<double>(i) + 0.5, 1.e-6));

      for (size_t j = 0; j < 3; j++) {
        REQUIRE_THAT(py[j], Catch::Matchers::WithinAbs(static_cast<double>(3 * i + j) + 0.01, 1.e-6));
      }
    }

    int8_t s1;
    float s2;
    int64_t s3;
    double s4;
    char *s5;

    std::memcpy(&s1, aosBuffer.get() + sizeof(SoA::Metadata::value_element) * elems, sizeof(int8_t));
    std::memcpy(&s2, aosBuffer.get() + sizeof(SoA::Metadata::value_element) * elems + sizeof(int8_t), sizeof(float));
    std::memcpy(&s3,
                aosBuffer.get() + sizeof(SoA::Metadata::value_element) * elems + sizeof(int8_t) + sizeof(float),
                sizeof(int64_t));
    std::memcpy(&s4,
                aosBuffer.get() + sizeof(SoA::Metadata::value_element) * elems + sizeof(int8_t) + sizeof(float) +
                    sizeof(int64_t),
                sizeof(double));
    std::memcpy(&s5,
                aosBuffer.get() + sizeof(SoA::Metadata::value_element) * elems + sizeof(int8_t) + sizeof(float) +
                    sizeof(int64_t) + sizeof(double),
                sizeof(const char *));

    REQUIRE(s1 == 100);
    REQUIRE_THAT(s2, Catch::Matchers::WithinAbs(42.42f, 1.e-6));
    REQUIRE(s3 == (int64_t(1) << 42) + 852516352);
    REQUIRE_THAT(s4, Catch::Matchers::WithinAbs(static_cast<double>((int64_t(1) << 42) + 8.52516352), 1.e-6));
    REQUIRE(std::string(s5) == "Testing");

    // check that we can go back from AoS to SoA
    std::unique_ptr<std::byte, decltype(std::free) *> soaBuffer2{
        reinterpret_cast<std::byte *>(aligned_alloc(SoA::alignment, soaBufferSize)), std::free};

    SoA soa2{soaBuffer2.get(), elems};
    SoAView soaView2{soa2};
    SoAConstView soaConstView2{soa2};

    for (size_t i = 0; i < elems; i++)
      soaView2.transpose(aosView, i);

    for (size_t i = 0; i < elems; i++) {
      REQUIRE_THAT(soaConstView2[i].xPos(), Catch::Matchers::WithinAbs(static_cast<float>(i), 1.e-6));
      REQUIRE_THAT(soaConstView2[i].yPos(), Catch::Matchers::WithinAbs(static_cast<float>(i) + 0.1f, 1.e-6));
      REQUIRE_THAT(soaConstView2[i].zPos(), Catch::Matchers::WithinAbs(static_cast<float>(i) + 0.2f, 1.e-6));
      REQUIRE_THAT(soaConstView2[i].candidateDirection()(0),
                   Catch::Matchers::WithinAbs(static_cast<double>(i) + 0.3, 1.e-6));
      REQUIRE_THAT(soaConstView2[i].candidateDirection()(1),
                   Catch::Matchers::WithinAbs(static_cast<double>(i) + 0.4, 1.e-6));
      REQUIRE_THAT(soaConstView2[i].candidateDirection()(2),
                   Catch::Matchers::WithinAbs(static_cast<double>(i) + 0.5, 1.e-6));

      const double *py = soaConstView2[i].py();
      for (size_t j = 0; j < 3; j++) {
        REQUIRE_THAT(py[j], Catch::Matchers::WithinAbs(static_cast<double>(3 * i + j) + 0.01, 1.e-6));
      }
    }

    REQUIRE(soaConstView2.s1() == 100);
    REQUIRE_THAT(soaConstView2.s2(), Catch::Matchers::WithinAbs(42.42f, 1.e-6));
    REQUIRE(soaConstView2.s3() == (int64_t(1) << 42) + 852516352);
    REQUIRE_THAT(soaConstView2.s4(),
                 Catch::Matchers::WithinAbs(static_cast<double>((int64_t(1) << 42) + 8.52516352), 1.e-6));
    REQUIRE(std::string(soaConstView2.s5()) == "Testing");
  }

  SECTION("Test for SoATemplateOnlyScalars") {
    const std::size_t soaBufferSize = SoAOnlyScalars::computeDataSize(elems);
    const std::size_t aosBufferSize = SoAOnlyScalars::AoSWrapper::computeDataSize(elems);
    // The AoS buffer is just the size of the scalar members
    // Size of an empty struct is 1 byte!
    const std::size_t expectedBufferSize = elems + sizeof(int8_t) + sizeof(float) + sizeof(int64_t) + sizeof(double);
    REQUIRE(sizeof(SoAOnlyScalars::Metadata::value_element) == 1);
    REQUIRE(expectedBufferSize == aosBufferSize);

    // memory buffer for the SoA of positions
    std::unique_ptr<std::byte, decltype(std::free) *> soaBuffer{
        reinterpret_cast<std::byte *>(aligned_alloc(SoA::alignment, soaBufferSize)), std::free};

    std::unique_ptr<std::byte, decltype(std::free) *> aosBuffer{
        reinterpret_cast<std::byte *>(std::malloc(aosBufferSize)), std::free};

    // SoA Layout
    SoAOnlyScalars soa{soaBuffer.get(), elems};

    // SoA Views
    SoAViewOnlyScalar soaView{soa};
    SoAConstViewOnlyScalar soaConstView{soa};

    soaView.s1() = 100;
    soaView.s2() = 42.42f;
    soaView.s3() = (int64_t(1) << 42) + 852516352;
    soaView.s4() = static_cast<double>((int64_t(1) << 42) + 8.52516352);

    SoAOnlyScalars::AoSWrapper aos{aosBuffer.get(), elems};
    SoAOnlyScalars::AoSWrapper::View aosView{aos};
    SoAOnlyScalars::AoSWrapper::ConstView aosConstView{aos};

    for (size_t i = 0; i < elems; i++)
      aosView.transpose(soaView, i);

    REQUIRE(aosConstView.s1() == 100);
    REQUIRE_THAT(aosConstView.s2(), Catch::Matchers::WithinAbs(42.42f, 1.e-6));
    REQUIRE(aosConstView.s3() == (int64_t(1) << 42) + 852516352);
    REQUIRE_THAT(aosConstView.s4(),
                 Catch::Matchers::WithinAbs(static_cast<double>((int64_t(1) << 42) + 8.52516352), 1.e-6));
  }
}
