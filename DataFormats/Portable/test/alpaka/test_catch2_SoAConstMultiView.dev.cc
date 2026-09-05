#include <Eigen/Core>
#include <Eigen/Dense>

#include <alpaka/alpaka.hpp>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/SoATemplate/interface/SoAConstMultiView.h"

#include "DataFormats/Portable/interface/PortableCollection.h"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
using namespace Catch::Matchers;

GENERATE_SOA_LAYOUT(SoAPositionTemplate,
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z),
                    SOA_SCALAR(int, s1),
                    SOA_SCALAR(float, s2))

using SoAPosition = SoAPositionTemplate<>;
using SoAPositionView = SoAPosition::View;
using SoAPositionConstView = SoAPosition::ConstView;
using SoAPositionMultiView = SoAConstMultiView<SoAPositionConstView, 5>;

GENERATE_SOA_LAYOUT(SoAPCATemplate,
                    SOA_COLUMN(float, vector_1),
                    SOA_COLUMN(float, vector_2),
                    SOA_COLUMN(float, vector_3),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, candidateDirection))

using SoAPCA = SoAPCATemplate<>;
using SoAPCAView = SoAPCA::View;
using SoAPCAConstView = SoAPCA::ConstView;
using SoAPCAMultiView = SoAConstMultiView<SoAPCAConstView, 5>;

GENERATE_SOA_BLOCKS(SoABlocksTemplate, SOA_BLOCK(position, SoAPositionTemplate), SOA_BLOCK(pca, SoAPCATemplate))

using SoA = SoABlocksTemplate<>;
using SoAView = SoA::View;
using SoAConstView = SoA::ConstView;

struct checkPositionMultiView {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, SoAPositionMultiView view, float* output) const {
    for (auto i : cms::alpakatools::uniform_elements(acc, view.size())) {
      // For s1 we take the sum of all s1 values in the view, for s2 we take the value from the first view
      int s1 = 0;
      for (int j = 0; j < view.numViews(); ++j) {
        s1 += view.view(j).s1();
      }
      const float s2 = view.view(0).s2();

      auto si = view[i];
      output[i] = si.x() * si.x() + si.y() * si.y() + si.z() * si.z() + static_cast<float>(s1) + s2;
    }
  }
};

struct checkPCAMultiView {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, SoAPCAMultiView view, float* output) const {
    for (auto i : cms::alpakatools::uniform_elements(acc, view.size())) {
      auto si = view[i];
      output[i] = si.vector_1() * si.vector_1() + si.vector_2() * si.vector_2() + si.vector_3() * si.vector_3() +
                  static_cast<float>(si.candidateDirection().squaredNorm());
    }
  }
};

TEST_CASE("PortableSoAConstMultiView") {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cout << "No devices available for the " << EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE)
              << " backend, skipping.\n";
    return;
  }

  for (auto const& device : devices) {
    std::cout << "Running on " << alpaka::getName(device) << std::endl;
    Queue queue(device);

    std::array<cms::soa::size_type, 2> sizes1{{42, 69}};

    PortableHostCollection<SoA> hostCollection1(cms::alpakatools::host(), sizes1);
    auto h_view1 = hostCollection1.view();

    // fill up
    for (cms::soa::size_type i = 0; i < sizes1[0]; i++) {
      h_view1.position()[i].x() = static_cast<float>(i);
      h_view1.position()[i].y() = static_cast<float>(i) * 2.0f;
      h_view1.position()[i].z() = static_cast<float>(i) * 3.0f;
    }
    h_view1.position().s1() = 21;
    h_view1.position().s2() = 21.23;
    for (cms::soa::size_type i = 0; i < sizes1[1]; i++) {
      h_view1.pca()[i].vector_1() = static_cast<float>(i);
      h_view1.pca()[i].vector_2() = static_cast<float>(i) * 2.0f;
      h_view1.pca()[i].vector_3() = static_cast<float>(i) * 3.0f;
      h_view1.pca()[i].candidateDirection() = Eigen::Vector3d(i, i * 2.0, i * 3.0);
    }

    std::array<cms::soa::size_type, 2> sizes2{{420, 666}};
    PortableHostCollection<SoA> hostCollection2(cms::alpakatools::host(), sizes2);
    auto h_view2 = hostCollection2.view();

    // fill up
    for (cms::soa::size_type i = 0; i < sizes2[0]; i++) {
      h_view2.position()[i].x() = static_cast<float>(i) * 10.0f;
      h_view2.position()[i].y() = static_cast<float>(i) * 11.0f;
      h_view2.position()[i].z() = static_cast<float>(i) * 12.0f;
    }
    h_view2.position().s1() = 42;
    h_view2.position().s2() = 42.43;
    for (cms::soa::size_type i = 0; i < sizes2[1]; i++) {
      h_view2.pca()[i].vector_1() = static_cast<float>(i) * 17.0f;
      h_view2.pca()[i].vector_2() = static_cast<float>(i) * 18.0f;
      h_view2.pca()[i].vector_3() = static_cast<float>(i) * 19.0f;
      h_view2.pca()[i].candidateDirection() = Eigen::Vector3d(i * 111.0, i * 222.0, i * 333.0);
    }

    // for the position multi view we restrict the iteration range for both views
    std::vector<int> offsetsPositionMultiView{sizes1[0] / 3, sizes1[1] / 2};

    std::vector<std::reference_wrapper<const PortableHostCollection<SoA>>> hostCollections;
    hostCollections.emplace_back(hostCollection1);
    hostCollections.emplace_back(hostCollection2);

    SoAPositionMultiView hostPositionMultiView(
        hostCollections,
        [](auto const& collection) { return collection.get().const_view().position(); },
        offsetsPositionMultiView);

    PortableCollection<Device, SoA> deviceCollection1(queue, sizes1);
    alpaka::memcpy(queue, deviceCollection1.buffer(), hostCollection1.buffer());
    PortableCollection<Device, SoA> deviceCollection2(queue, sizes2);
    alpaka::memcpy(queue, deviceCollection2.buffer(), hostCollection2.buffer());

    std::vector<std::reference_wrapper<const PortableCollection<Device, SoA>>> deviceCollections;
    deviceCollections.emplace_back(deviceCollection1);
    deviceCollections.emplace_back(deviceCollection2);

    SoAPositionMultiView positionMultiView(
        deviceCollections,
        [](auto const& collection) { return collection.get().const_view().position(); },
        offsetsPositionMultiView);
    SoAPCAMultiView pcaMultiView(deviceCollections,
                                 [](auto const& collection) { return collection.get().const_view().pca(); });

    REQUIRE(positionMultiView.size() == offsetsPositionMultiView[0] + offsetsPositionMultiView[1]);
    REQUIRE(pcaMultiView.size() == sizes1[1] + sizes2[1]);
    REQUIRE(hostPositionMultiView.size() == offsetsPositionMultiView[0] + offsetsPositionMultiView[1]);

    REQUIRE(positionMultiView.numViews() == 2);
    REQUIRE(pcaMultiView.numViews() == 2);
    REQUIRE(hostPositionMultiView.numViews() == 2);

    auto resultPosition_d = cms::alpakatools::make_device_buffer<float[]>(queue, positionMultiView.size());
    auto resultPCA_d = cms::alpakatools::make_device_buffer<float[]>(queue, pcaMultiView.size());
    auto resultPosition_h = cms::alpakatools::make_host_buffer<float[]>(queue, positionMultiView.size());
    auto resultPCA_h = cms::alpakatools::make_host_buffer<float[]>(queue, pcaMultiView.size());
    alpaka::wait(queue);

    const std::size_t blockSize = 64;

    const std::size_t nBlocksPositionKernel = cms::alpakatools::divide_up_by(positionMultiView.size(), blockSize);
    const auto workDivPositionKernel = cms::alpakatools::make_workdiv<Acc1D>(nBlocksPositionKernel, blockSize);
    const std::size_t nBlocksPCAKernel = cms::alpakatools::divide_up_by(pcaMultiView.size(), blockSize);
    const auto workDivPCAKernel = cms::alpakatools::make_workdiv<Acc1D>(nBlocksPCAKernel, blockSize);

    alpaka::exec<Acc1D>(
        queue, workDivPositionKernel, checkPositionMultiView{}, positionMultiView, resultPosition_d.data());
    alpaka::exec<Acc1D>(queue, workDivPCAKernel, checkPCAMultiView{}, pcaMultiView, resultPCA_d.data());
    alpaka::wait(queue);

    alpaka::memcpy(queue, hostCollection1.buffer(), deviceCollection1.buffer());
    alpaka::memcpy(queue, hostCollection2.buffer(), deviceCollection2.buffer());

    alpaka::memcpy(queue, resultPosition_h, resultPosition_d);
    alpaka::memcpy(queue, resultPCA_h, resultPCA_d);

    alpaka::wait(queue);

    // check results
    for (cms::soa::size_type i = 0; i < hostPositionMultiView.size(); ++i) {
      int s1 = 0;
      for (int j = 0; j < hostPositionMultiView.numViews(); ++j) {
        s1 += hostPositionMultiView.view(j).s1();
      }
      auto const s2 = hostPositionMultiView.view(0).s2();
      auto si = hostPositionMultiView[i];
      const float expected = si.x() * si.x() + si.y() * si.y() + si.z() * si.z() + static_cast<float>(s1) + s2;
      REQUIRE(resultPosition_h[i] == Catch::Approx(expected).margin(1e-5));
    }

    // check results
    for (cms::soa::size_type i = 0; i < pcaMultiView.size(); ++i) {
      auto si = i < sizes1[1] ? h_view1.pca()[i] : h_view2.pca()[i - sizes1[1]];
      const float expected = si.vector_1() * si.vector_1() + si.vector_2() * si.vector_2() +
                             si.vector_3() * si.vector_3() + static_cast<float>(si.candidateDirection().squaredNorm());
      REQUIRE(resultPCA_h[i] == Catch::Approx(expected).margin(1e-5));
    }
  }
}
