#include <Eigen/Core>
#include <Eigen/Dense>

#include <alpaka/alpaka.hpp>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/Portable/interface/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

using Vector5f = Eigen::Matrix<float, 5, 1>;
using Vector15f = Eigen::Matrix<float, 15, 1>;

GENERATE_SOA_LAYOUT(SoATemplate,
                    SOA_COLUMN(float, quality),
                    SOA_COLUMN(float, chi2),
                    SOA_COLUMN(int8_t, nLayers),
                    SOA_COLUMN(float, eta),
                    SOA_COLUMN(float, pt),
                    SOA_EIGEN_COLUMN(Vector5f, state),
                    SOA_EIGEN_COLUMN(Vector15f, covariance),
                    SOA_SCALAR(int, nTracks),
                    SOA_COLUMN(uint32_t, hitOffsets))

using SoA = SoATemplate<>;
using SoAView = SoA::View;
using SoAConstView = SoA::ConstView;

namespace {
  template <typename F, std::size_t... Is>
  void unrollColumns(F&& f, std::index_sequence<Is...>) {
    (f(std::integral_constant<std::size_t, Is>{}), ...);
  }

  template <std::size_t N, typename F>
  void mergeSoAColumns(F&& f) {
    unrollColumns(std::forward<F>(f), std::make_index_sequence<N>{});
  }
}  // namespace

struct SumScalar {
  template <typename T>
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, T* result, const T* v1, const T* v2) const {
    *result = *v1 + *v2;
  }
};

TEST_CASE("test merge soa alpaka", "[SoAMerge][Alpaka]") {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cout << "No devices available for the " << EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE)
              << " backend, skipping.\n";
    return;
  }

  for (auto const& device : devices) {
    std::cout << "Running on " << alpaka::getName(device) << std::endl;
    Queue queue(device);

    constexpr int n1 = 10;
    constexpr int n2 = 20;

    PortableHostCollection<SoA> hostCollection1(cms::alpakatools::host(), n1);
    auto h_view1 = hostCollection1.view();

    PortableHostCollection<SoA> hostCollection2(cms::alpakatools::host(), n2);
    auto h_view2 = hostCollection2.view();

    // fill up
    for (int i = 0; i < hostCollection1.size(); i++) {
      h_view1[i].quality() = static_cast<float>(1);
      h_view1[i].chi2() = static_cast<float>(2);
      h_view1[i].nLayers() = static_cast<int8_t>(3);
      h_view1[i].eta() = static_cast<float>(4);
      h_view1[i].pt() = static_cast<float>(5);
      h_view1[i].state().setConstant(6.f);
      h_view1[i].covariance().setConstant(7.f);
      h_view1[i].hitOffsets() = static_cast<uint32_t>(9);
    }
    h_view1.nTracks() = 8;

    for (int i = 0; i < hostCollection2.size(); i++) {
      h_view2[i].quality() = static_cast<float>(10);
      h_view2[i].chi2() = static_cast<float>(11);
      h_view2[i].nLayers() = static_cast<int8_t>(12);
      h_view2[i].eta() = static_cast<float>(13);
      h_view2[i].pt() = static_cast<float>(14);
      h_view2[i].state().setConstant(15.f);
      h_view2[i].covariance().setConstant(16.f);
      h_view2[i].hitOffsets() = static_cast<uint32_t>(18);
    }
    h_view2.nTracks() = 17;

    PortableCollection<Device, SoA> deviceCollection1(queue, hostCollection1.size());
    auto d_view1 = deviceCollection1.view();
    alpaka::memcpy(queue, deviceCollection1.buffer(), hostCollection1.buffer());

    PortableCollection<Device, SoA> deviceCollection2(queue, hostCollection2.size());
    auto d_view2 = deviceCollection2.view();
    alpaka::memcpy(queue, deviceCollection2.buffer(), hostCollection2.buffer());

    const int nTk1 = h_view1.nTracks();
    const int nTk2 = h_view2.nTracks();
    const int nTotal = nTk1 + nTk2;

    PortableCollection<Device, SoA> outputDevice(queue, nTk1 + nTk2);
    auto d_viewOut = outputDevice.view();

    alpaka::wait(queue);

    auto outDesc = SoA::Descriptor(d_viewOut);
    auto inDesc1 = SoA::ConstDescriptor(d_view1);
    auto inDesc2 = SoA::ConstDescriptor(d_view2);

    mergeSoAColumns<outDesc.num_cols>([&](auto columnIndex) {
      auto& outCol = std::get<columnIndex>(outDesc.buff);
      const auto& inCol1 = std::get<columnIndex>(inDesc1.buff);
      const auto& inCol2 = std::get<columnIndex>(inDesc2.buff);

      if constexpr (std::get<columnIndex>(outDesc.columnTypes) == cms::soa::SoAColumnType::scalar) {
        alpaka::exec<Acc1D>(queue,
                            cms::alpakatools::make_workdiv<Acc1D>(1, 1),
                            SumScalar{},
                            outCol.data(),
                            inCol1.data(),
                            inCol2.data());
      } else if constexpr (std::get<columnIndex>(outDesc.columnTypes) == cms::soa::SoAColumnType::eigen) {
        using EigenType = std::tuple_element_t<columnIndex, decltype(outDesc.parameterTypes)>::ValueType;
        constexpr int num_rows = EigenType::RowsAtCompileTime;

        const auto strideOutput = std::get<1>(std::get<columnIndex>(outDesc.parameterTypes).tupleOrPointer());
        const auto strideInput1 = std::get<1>(std::get<columnIndex>(inDesc1.parameterTypes).tupleOrPointer());
        const auto strideInput2 = std::get<1>(std::get<columnIndex>(inDesc2.parameterTypes).tupleOrPointer());

        for (int i = 0; i < num_rows; ++i) {
          const auto offsetOutput = i * strideOutput;
          const auto offsetIn1 = i * strideInput1;
          const auto offsetIn2 = i * strideInput2;
          alpaka::memcpy(queue,
                         cms::alpakatools::make_device_view(queue, outCol.data() + offsetOutput, nTk1),
                         cms::alpakatools::make_device_view(queue, inCol1.data() + offsetIn1, nTk1));
          // copy second collection with offset of first collection size
          alpaka::memcpy(queue,
                         cms::alpakatools::make_device_view(queue, outCol.data() + nTk1 + offsetOutput, nTk2),
                         cms::alpakatools::make_device_view(queue, inCol2.data() + offsetIn2, nTk2));
        }
      } else {
        alpaka::memcpy(queue,
                       cms::alpakatools::make_device_view(queue, outCol.data(), nTk1),
                       cms::alpakatools::make_device_view(queue, inCol1.data(), nTk1));
        // copy second collection with offset of first collection size
        alpaka::memcpy(queue,
                       cms::alpakatools::make_device_view(queue, outCol.data() + nTk1, nTk2),
                       cms::alpakatools::make_device_view(queue, inCol2.data(), nTk2));
      }
    });

    alpaka::wait(queue);

    PortableHostCollection<SoA> outputHost(cms::alpakatools::host(), nTk1 + nTk2);
    auto h_viewOut = outputHost.view();
    alpaka::memcpy(queue, outputHost.buffer(), outputDevice.buffer());

    alpaka::wait(queue);

    REQUIRE(h_viewOut.nTracks() == nTotal);

    for (int i = 0; i < nTk1 + nTk2; i++) {
      if (i < nTk1) {
        REQUIRE(h_viewOut[i].quality() == Catch::Approx(1.0f));
        REQUIRE(h_viewOut[i].chi2() == Catch::Approx(2.0f));
        REQUIRE(h_viewOut[i].nLayers() == 3);
        REQUIRE(h_viewOut[i].eta() == Catch::Approx(4.0f));
        REQUIRE(h_viewOut[i].pt() == Catch::Approx(5.0f));
        REQUIRE(h_viewOut[i].state() == Vector5f(6.f, 6.f, 6.f, 6.f, 6.f));
        REQUIRE(h_viewOut[i].covariance() ==
                Vector15f(7.f, 7.f, 7.f, 7.f, 7.f, 7.f, 7.f, 7.f, 7.f, 7.f, 7.f, 7.f, 7.f, 7.f, 7.f));
      } else {
        REQUIRE(h_viewOut[i].quality() == Catch::Approx(10.0f));
        REQUIRE(h_viewOut[i].chi2() == Catch::Approx(11.0f));
        REQUIRE(h_viewOut[i].nLayers() == 12);
        REQUIRE(h_viewOut[i].eta() == Catch::Approx(13.0f));
        REQUIRE(h_viewOut[i].pt() == Catch::Approx(14.0f));
        REQUIRE(h_viewOut[i].state() == Vector5f(15.f, 15.f, 15.f, 15.f, 15.f));
        REQUIRE(h_viewOut[i].covariance() ==
                Vector15f(16.f, 16.f, 16.f, 16.f, 16.f, 16.f, 16.f, 16.f, 16.f, 16.f, 16.f, 16.f, 16.f, 16.f, 16.f));
      }
    }
  }
}
