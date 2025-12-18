#include <alpaka/alpaka.hpp>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/Portable/interface/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

GENERATE_SOA_LAYOUT(SoATemplate,
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z),
                    SOA_COLUMN(double, v_x),
                    SOA_COLUMN(double, v_y),
                    SOA_COLUMN(double, v_z),

                    SOA_ELEMENT_METHODS(

                        SOA_HOST_DEVICE void normalise() {
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
                        SOA_HOST_DEVICE float square_norm_position()
                            const { return sqrt(x() * x() + y() * y() + z() * z()); };

                        SOA_HOST_DEVICE double square_norm_velocity()
                            const { return sqrt(v_x() * v_x() + v_y() * v_y() + v_z() * v_z()); };

                        template <typename T1, typename T2>
                        SOA_HOST_DEVICE static auto time(T1 pos, T2 vel) {
                          if (not(vel == 0))
                            return pos / vel;
                          return 0.;
                        }),

                    SOA_SCALAR(int, detectorType))

using SoA = SoATemplate<>;
using SoAView = SoA::View;
using SoAConstView = SoA::ConstView;

GENERATE_SOA_LAYOUT(ResultTemplate,
                    SOA_COLUMN(float, positionNorm),
                    SOA_COLUMN(double, velocityNorm),
                    SOA_COLUMN(double, times))

using ResultSoA = ResultTemplate<>;
using ResultView = ResultSoA::View;

struct calculateNorm {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, SoAConstView soaConstView, ResultView resultView) const {
    for (auto i : cms::alpakatools::uniform_elements(acc, soaConstView.metadata().size())) {
      resultView[i].positionNorm() = soaConstView[i].square_norm_position();
      resultView[i].velocityNorm() = soaConstView[i].square_norm_velocity();
    }
  }
};

struct checkNormalise {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, SoAView soaView, ResultView resultView) const {
    for (auto i : cms::alpakatools::uniform_elements(acc, soaView.metadata().size())) {
      resultView[i].times() = SoAView::const_element::time(soaView[i].x(), soaView[i].v_x());
      soaView[i].normalise();
    }
  }
};

TEST_CASE("SoACustomizedMethods Alpaka", "[SoACustomizedMethods][Alpaka]") {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cout << "No devices available for the " << EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE)
              << " backend, skipping.\n";
    return;
  }

  for (auto const& device : devices) {
    std::cout << "Running on " << alpaka::getName(device) << std::endl;
    Queue queue(device);

    constexpr unsigned int elems = 10;

    PortableHostCollection<SoA> hostCollection(elems, cms::alpakatools::host());
    auto h_view = hostCollection.view();
    const auto h_Constview = hostCollection.const_view();

    PortableHostCollection<ResultSoA> hostResultCollection(elems, cms::alpakatools::host());
    auto h_result_view = hostResultCollection.view();

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

    PortableCollection<SoA, Device> deviceCollection(elems, queue);
    auto d_view = deviceCollection.view();
    auto d_Constview = deviceCollection.const_view();
    alpaka::memcpy(queue, deviceCollection.buffer(), hostCollection.buffer());

    PortableCollection<ResultSoA, Device> deviceResultCollection(elems, queue);
    auto d_result_view = deviceResultCollection.view();
    alpaka::wait(queue);

    // Work division
    const std::size_t blockSize = 256;
    const std::size_t numberOfBlocks = cms::alpakatools::divide_up_by(elems, blockSize);
    const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

    SECTION("ConstView methods Alpaka") {
      alpaka::exec<Acc1D>(queue, workDiv, calculateNorm{}, d_Constview, d_result_view);
      alpaka::wait(queue);

      alpaka::memcpy(queue, hostResultCollection.buffer(), deviceResultCollection.buffer());

      // Check for the correctness of the square_norm() functions
      for (size_t i = 0; i < elems; i++) {
        const float position_norm =
            sqrt(h_Constview[i].x() * h_Constview[i].x() + h_Constview[i].y() * h_Constview[i].y() +
                 h_Constview[i].z() * h_Constview[i].z());
        const double velocity_norm =
            sqrt(h_Constview[i].v_x() * h_Constview[i].v_x() + h_Constview[i].v_y() * h_Constview[i].v_y() +
                 h_Constview[i].v_z() * h_Constview[i].v_z());
        REQUIRE(h_result_view[i].positionNorm() == position_norm);
        REQUIRE(h_result_view[i].velocityNorm() == velocity_norm);
      }
    }

    SECTION("View methods Alpaka") {
      std::array<double, elems> times;

      // Check for the correctness of the time() function
      times[0] = 0.;
      for (size_t i = 0; i < elems; i++) {
        if (!(i == 0))
          times[i] = h_view[i].x() / h_view[i].v_x();
      }

      alpaka::exec<Acc1D>(queue, workDiv, checkNormalise{}, d_view, d_result_view);
      alpaka::wait(queue);

      alpaka::memcpy(queue, hostResultCollection.buffer(), deviceResultCollection.buffer());
      alpaka::memcpy(queue, hostCollection.buffer(), deviceCollection.buffer());

      // Check for the correctness of the time() function
      for (size_t i = 0; i < elems; i++) {
        REQUIRE(h_result_view[i].times() == times[i]);
      }

      REQUIRE(h_view[0].square_norm_position() == 0.f);
      REQUIRE(h_view[0].square_norm_velocity() == 0.);
      for (size_t i = 1; i < elems; i++) {
        REQUIRE_THAT(h_view[i].square_norm_position(), Catch::Matchers::WithinAbs(1.f, 1.e-6));
        REQUIRE_THAT(h_view[i].square_norm_velocity(), Catch::Matchers::WithinAbs(1., 1.e-9));
      }
    }
  }
}
