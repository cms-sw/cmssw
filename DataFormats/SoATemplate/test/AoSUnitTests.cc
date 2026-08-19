#include <Eigen/Core>
#include <Eigen/Dense>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include <iostream>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(SoATemplate,
                    SOA_SCALAR(int8_t, s1),
                    SOA_COLUMN(float, f1),
                    SOA_COLUMN(float, f2),
                    SOA_COLUMN(int8_t, i1),
                    SOA_SCALAR(float, s2),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, candidateDirection),
                    SOA_COLUMN(int64_t, i2),
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

TEST_CASE("AoS Unit Tests") {
  // common number of elements for the SoAs
  const SoA::size_type elems = 16;
  const auto soaBufferSize = SoA::computeDataSize(elems);
  const auto aosBufferSize = SoA::AoSWrapper::computeDataSize(elems);
  // The AoS is an array of SoA::Metadata::value_element
  // So the total memory is sizeof(SoA::Metadata::value_element) * elems + the size of the scalar members
  const auto expectedBufferSize = sizeof(SoA::Metadata::value_element) * elems + sizeof(int8_t) + sizeof(float) +
                                  sizeof(int64_t) + sizeof(double) + sizeof(const char *);
  REQUIRE(expectedBufferSize == aosBufferSize);

  // memory buffer for the SoA
  std::unique_ptr<std::byte, decltype(std::free) *> soaBuffer{
      reinterpret_cast<std::byte *>(aligned_alloc(SoA::alignment, soaBufferSize)), std::free};

  std::unique_ptr<std::byte, decltype(std::free) *> aosBuffer{reinterpret_cast<std::byte *>(std::malloc(aosBufferSize)),
                                                              std::free};

  // SoA Layout
  SoA soa{soaBuffer.get(), elems};

  // SoA Views
  SoAView soaView{soa};
  SoAConstView soaConstView{soa};

  // AoS Layout and Views
  SoA::AoSWrapper aos{aosBuffer.get(), elems};
  SoA::AoSWrapper::View aosView{aos};
  SoA::AoSWrapper::ConstView aosConstView{aos};

  // fill up the SoA Layout
  for (size_t i = 0; i < elems; i++) {
    soaView[i].f1() = static_cast<float>(i);
    soaView[i].f2() = static_cast<float>(i) + 0.1f;

    soaView[i].i1() = static_cast<int8_t>(i);

    soaView[i].candidateDirection()(0) = static_cast<double>(i) + 0.3;
    soaView[i].candidateDirection()(1) = static_cast<double>(i) + 0.4;
    soaView[i].candidateDirection()(2) = static_cast<double>(i) + 0.5;

    soaView[i].i2() = static_cast<int64_t>(i) * 4269420666;
  }
  soaView.s1() = 100;
  soaView.s2() = 42.42f;
  soaView.s3() = (int64_t(1) << 42) + 852516352;
  soaView.s4() = static_cast<double>((int64_t(1) << 42) + 8.52516352);
  soaView.s5() = "Testing";

  // Copy to AoS
  for (size_t i = 0; i < elems; i++) {
    aosView.transpose(soaConstView, i);
  }

  SECTION("AoS test basic functionality") {
    // Check that the data is the same in the SoA and AoS views
    REQUIRE(soaConstView.metadata().size() == aosConstView.metadata().size());
    REQUIRE(elems == aosConstView.metadata().size());

    for (size_t i = 0; i < elems; i++) {
      auto element = aosConstView[i];
      // check that all values match
      REQUIRE_THAT(element.f1(), Catch::Matchers::WithinAbs(static_cast<float>(i), 1.e-6));
      REQUIRE_THAT(element.f2(), Catch::Matchers::WithinAbs(static_cast<float>(i) + 0.1f, 1.e-6));

      REQUIRE(element.i1() == static_cast<int8_t>(i));

      REQUIRE_THAT(element.candidateDirection()(0),
                   Catch::Matchers::WithinAbs(static_cast<double>(i) + 0.3, 1.e-6));
      REQUIRE_THAT(element.candidateDirection()(1),
                   Catch::Matchers::WithinAbs(static_cast<double>(i) + 0.4, 1.e-6));
      REQUIRE_THAT(element.candidateDirection()(2),
                   Catch::Matchers::WithinAbs(static_cast<double>(i) + 0.5, 1.e-6));

      REQUIRE(element.i2() == static_cast<int64_t>(i) * 4269420666);

      // check that alternative accessors work as well
      REQUIRE_THAT(aosConstView.f1(i), Catch::Matchers::WithinAbs(element.f1(), 1.e-6));
      REQUIRE_THAT(aosConstView.f1()[i], Catch::Matchers::WithinAbs(element.f1(), 1.e-6));
      REQUIRE_THAT(aosConstView.f2(i), Catch::Matchers::WithinAbs(element.f2(), 1.e-6));
      REQUIRE_THAT(aosConstView.f2()[i], Catch::Matchers::WithinAbs(element.f2(), 1.e-6));
      REQUIRE(aosConstView.i1(i) == element.i1());
      REQUIRE(aosConstView.i1()[i] == element.i1());
      REQUIRE_THAT(aosConstView.candidateDirection(i)(0),
                   Catch::Matchers::WithinAbs(element.candidateDirection()(0), 1.e-6));
      REQUIRE_THAT(aosConstView.candidateDirection(i)(1),
                   Catch::Matchers::WithinAbs(element.candidateDirection()(1), 1.e-6));
      REQUIRE_THAT(aosConstView.candidateDirection(i)(2),
                   Catch::Matchers::WithinAbs(element.candidateDirection()(2), 1.e-6));
      REQUIRE_THAT(aosConstView.candidateDirection()[i](0),
                   Catch::Matchers::WithinAbs(element.candidateDirection()(0), 1.e-6));
      REQUIRE_THAT(aosConstView.candidateDirection()[i](1),
                   Catch::Matchers::WithinAbs(element.candidateDirection()(1), 1.e-6));
      REQUIRE_THAT(aosConstView.candidateDirection()[i](2),
                   Catch::Matchers::WithinAbs(element.candidateDirection()(2), 1.e-6));
      REQUIRE(aosConstView.i2(i) == element.i2());
      REQUIRE(aosConstView.i2()[i] == element.i2());
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

    REQUIRE_THROWS_AS(aosConstView.f1(underflow), std::out_of_range);
    REQUIRE_THROWS_AS(aosConstView.f1(overflow), std::out_of_range);
    REQUIRE_THROWS_AS(aosConstView.f2(underflow), std::out_of_range);
    REQUIRE_THROWS_AS(aosConstView.f2(overflow), std::out_of_range);
    REQUIRE_THROWS_AS(aosConstView.i1(underflow), std::out_of_range);
    REQUIRE_THROWS_AS(aosConstView.i1(overflow), std::out_of_range);
    REQUIRE_THROWS_AS(aosConstView.i2(underflow), std::out_of_range);
    REQUIRE_THROWS_AS(aosConstView.i2(overflow), std::out_of_range);
    REQUIRE_THROWS_AS(aosConstView.candidateDirection(underflow), std::out_of_range);
    REQUIRE_THROWS_AS(aosConstView.candidateDirection(overflow), std::out_of_range);


    // Check for under-and overflow in the row accessor
    REQUIRE_THROWS_AS(aosView[underflow], std::out_of_range);
    REQUIRE_THROWS_AS(aosView[overflow], std::out_of_range);
    REQUIRE_THROWS_AS(aosView.f1(underflow), std::out_of_range);
    REQUIRE_THROWS_AS(aosView.f1(overflow), std::out_of_range);
    REQUIRE_THROWS_AS(aosView.f2(underflow), std::out_of_range);
    REQUIRE_THROWS_AS(aosView.f2(overflow), std::out_of_range);
    REQUIRE_THROWS_AS(aosView.i1(underflow), std::out_of_range);
    REQUIRE_THROWS_AS(aosView.i1(overflow), std::out_of_range);
    REQUIRE_THROWS_AS(aosView.i2(underflow), std::out_of_range);
    REQUIRE_THROWS_AS(aosView.i2(overflow), std::out_of_range);
    REQUIRE_THROWS_AS(aosView.candidateDirection(underflow), std::out_of_range);
    REQUIRE_THROWS_AS(aosView.candidateDirection(overflow), std::out_of_range);
  }

  SECTION("AoS test memory layout") {
    // Check that the AoS memory layout is as expected
    const auto stride = sizeof(SoA::Metadata::value_element);
    for (size_t i = 0; i < elems; i++) {
      float f1;
      float f2;

      int8_t i1;

      double candidateDirection0;
      double candidateDirection1;
      double candidateDirection2;

      int64_t i2;

      std::memcpy(&f1, aosBuffer.get() + offsetof(SoA::Metadata::value_element, f1_) + i * stride, sizeof(float));
      std::memcpy(&f2, aosBuffer.get() + offsetof(SoA::Metadata::value_element, f2_) + i * stride, sizeof(float));

      std::memcpy(&i1, aosBuffer.get() + offsetof(SoA::Metadata::value_element, i1_) + i * stride, sizeof(int8_t));

      const auto offsetCandidateDirection = offsetof(SoA::Metadata::value_element, candidateDirection_) + i * stride;
      std::memcpy(&candidateDirection0, aosBuffer.get() + offsetCandidateDirection, sizeof(double));
      std::memcpy(&candidateDirection1, aosBuffer.get() + offsetCandidateDirection + 8, sizeof(double));
      std::memcpy(&candidateDirection2, aosBuffer.get() + offsetCandidateDirection + 16, sizeof(double));

      std::memcpy(&i2, aosBuffer.get() + offsetof(SoA::Metadata::value_element, i2_) + i * stride, sizeof(int64_t));

      REQUIRE_THAT(f1, Catch::Matchers::WithinAbs(static_cast<float>(i), 1.e-6));
      REQUIRE_THAT(f2, Catch::Matchers::WithinAbs(static_cast<float>(i) + 0.1f, 1.e-6));

      REQUIRE(i1 == static_cast<int8_t>(i));

      REQUIRE_THAT(candidateDirection0, Catch::Matchers::WithinAbs(static_cast<double>(i) + 0.3, 1.e-6));
      REQUIRE_THAT(candidateDirection1, Catch::Matchers::WithinAbs(static_cast<double>(i) + 0.4, 1.e-6));
      REQUIRE_THAT(candidateDirection2, Catch::Matchers::WithinAbs(static_cast<double>(i) + 0.5, 1.e-6));

      REQUIRE(i2 ==static_cast<int64_t>(i) * 4269420666);
    }

    // Scalar values are appended at the end of the AoS buffer, this is checked here
    int8_t s1;
    float s2;
    int64_t s3;
    double s4;
    char *s5;

    const auto offsetScalars = sizeof(SoA::Metadata::value_element) * elems;
    std::memcpy(&s1, aosBuffer.get() + offsetScalars, sizeof(s1));
    std::memcpy(&s2, aosBuffer.get() + offsetScalars + sizeof(s1), sizeof(s2));
    std::memcpy(&s3, aosBuffer.get() + offsetScalars + sizeof(s1) + sizeof(s2), sizeof(s3));
    std::memcpy(&s4, aosBuffer.get() + offsetScalars + sizeof(s1) + sizeof(s2) + sizeof(s3), sizeof(s4));
    std::memcpy(&s5, aosBuffer.get() + offsetScalars + sizeof(s1) + sizeof(s2) + sizeof(s3) + sizeof(s4), sizeof(s5));

    REQUIRE(s1 == 100);
    REQUIRE_THAT(s2, Catch::Matchers::WithinAbs(42.42f, 1.e-6));
    REQUIRE(s3 == (int64_t(1) << 42) + 852516352);
    REQUIRE_THAT(s4, Catch::Matchers::WithinAbs(static_cast<double>((int64_t(1) << 42) + 8.52516352), 1.e-6));
    REQUIRE(std::string(s5) == "Testing");
  }

  SECTION("AoS test transpose to SoA") {
    // check that we can go back from AoS to SoA
    std::unique_ptr<std::byte, decltype(std::free) *> soaBuffer2{
        reinterpret_cast<std::byte *>(aligned_alloc(SoA::alignment, soaBufferSize)), std::free};

    SoA soa2{soaBuffer2.get(), elems};
    SoAView soaView2{soa2};
    SoAConstView soaConstView2{soa2};

    for (size_t i = 0; i < elems; i++) {
      soaView2.transpose(aosConstView, i);
    }

    for (size_t i = 0; i < elems; i++) {
      REQUIRE_THAT(soaConstView2[i].f1(), Catch::Matchers::WithinAbs(static_cast<float>(i), 1.e-6));
      REQUIRE_THAT(soaConstView2[i].f2(), Catch::Matchers::WithinAbs(static_cast<float>(i) + 0.1f, 1.e-6));
      REQUIRE(soaConstView2[i].i1() == static_cast<int8_t>(i));
      REQUIRE_THAT(soaConstView2[i].candidateDirection()(0),
                   Catch::Matchers::WithinAbs(static_cast<double>(i) + 0.3, 1.e-6));
      REQUIRE_THAT(soaConstView2[i].candidateDirection()(1),
                   Catch::Matchers::WithinAbs(static_cast<double>(i) + 0.4, 1.e-6));
      REQUIRE_THAT(soaConstView2[i].candidateDirection()(2),
                   Catch::Matchers::WithinAbs(static_cast<double>(i) + 0.5, 1.e-6));

      REQUIRE(soaConstView2[i].i2() == static_cast<int64_t>(i) * 4269420666);
    }

    REQUIRE(soaConstView2.s1() == 100);
    REQUIRE_THAT(soaConstView2.s2(), Catch::Matchers::WithinAbs(42.42f, 1.e-6));
    REQUIRE(soaConstView2.s3() == (int64_t(1) << 42) + 852516352);
    REQUIRE_THAT(soaConstView2.s4(),
                 Catch::Matchers::WithinAbs(static_cast<double>((int64_t(1) << 42) + 8.52516352), 1.e-6));
    REQUIRE(std::string(soaConstView2.s5()) == "Testing");
  }
}

TEST_CASE("AoS Unit Tests Scalar only") {
  const SoA::size_type elems = 16;
  const auto soaBufferSize = SoAOnlyScalars::computeDataSize(elems);
  const auto aosBufferSize = SoAOnlyScalars::AoSWrapper::computeDataSize(elems);
  // The AoS buffer is just the size of the scalar members
  // Size of an empty struct is 1 byte!
  const auto expectedBufferSize = elems + sizeof(int8_t) + sizeof(float) + sizeof(int64_t) + sizeof(double);
  REQUIRE(sizeof(SoAOnlyScalars::Metadata::value_element) == 1);
  REQUIRE(expectedBufferSize == aosBufferSize);

  // memory buffer for the SoA of positions
  std::unique_ptr<std::byte, decltype(std::free) *> soaBuffer{
      reinterpret_cast<std::byte *>(aligned_alloc(SoA::alignment, soaBufferSize)), std::free};

  std::unique_ptr<std::byte, decltype(std::free) *> aosBuffer{reinterpret_cast<std::byte *>(std::malloc(aosBufferSize)),
                                                              std::free};

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

  for (size_t i = 0; i < elems; i++){
    aosView.transpose(soaConstView, i);
  }
    

  REQUIRE(aosConstView.s1() == 100);
  REQUIRE_THAT(aosConstView.s2(), Catch::Matchers::WithinAbs(42.42f, 1.e-6));
  REQUIRE(aosConstView.s3() == (int64_t(1) << 42) + 852516352);
  REQUIRE_THAT(aosConstView.s4(),
               Catch::Matchers::WithinAbs(static_cast<double>((int64_t(1) << 42) + 8.52516352), 1.e-6));
}
