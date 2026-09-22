#include <cstddef>
#include <iostream>
#include <numeric>
#include <vector>

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

constexpr int maxViews = 5;

GENERATE_SOA_LAYOUT(SoAPositionTemplate,
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z),
                    SOA_SCALAR(int, s1),
                    SOA_SCALAR(float, s2))

using SoAPosition = SoAPositionTemplate<>;
using SoAPositionView = SoAPosition::View;
using SoAPositionConstView = SoAPosition::ConstView;
using SoAPositionMultiView = SoAConstMultiView<SoAPositionConstView, maxViews>;

GENERATE_SOA_LAYOUT(SoAPCATemplate,
                    SOA_COLUMN(float, vector_1),
                    SOA_COLUMN(float, vector_2),
                    SOA_COLUMN(float, vector_3),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, candidateDirection))

using SoAPCA = SoAPCATemplate<>;
using SoAPCAView = SoAPCA::View;
using SoAPCAConstView = SoAPCA::ConstView;
using SoAPCAMultiView = SoAConstMultiView<SoAPCAConstView, maxViews>;

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

    for (int nCollections = 1; nCollections <= maxViews; ++nCollections) {
      // constexpr int nCollections = 4;
      std::vector<cms::soa::size_type> sizesPositionLayout(nCollections);
      std::vector<cms::soa::size_type> sizesPCALayout(nCollections);
      std::vector<int> pcaOffsets(nCollections);

      int pcaOffset = 0;
      for (int i = 0; i < nCollections; ++i) {
        sizesPositionLayout[i] = 10 * (i + 1);
        sizesPCALayout[i] = 5 * (i + 1);
        pcaOffset += sizesPCALayout[i];
        pcaOffsets[i] = pcaOffset;
      }

      std::vector<PortableHostCollection<SoA>> hostCollections;

      for (int i = 0; i < nCollections; ++i) {
        hostCollections.emplace_back(cms::alpakatools::host(), sizesPositionLayout[i], sizesPCALayout[i]);
      }

      int globalCounter = 0;
      for (int i = 0; i < nCollections; ++i) {
        auto h_view = hostCollections[i].view();

        for (cms::soa::size_type j = 0; j < sizesPositionLayout[i]; ++j) {
          float val = static_cast<float>(globalCounter++);
          h_view.position()[j].x() = val;
          h_view.position()[j].y() = val + 0.1f;
          h_view.position()[j].z() = val + 0.2f;
        }

        h_view.position().s1() = i + 1;
        h_view.position().s2() = static_cast<float>(i + 1) * 1.23f;

        for (cms::soa::size_type j = 0; j < sizesPCALayout[i]; ++j) {
          float val = static_cast<float>(globalCounter++);
          h_view.pca()[j].vector_1() = val + 0.3f;
          h_view.pca()[j].vector_2() = val + 0.4f;
          h_view.pca()[j].vector_3() = val + 0.5f;

          double dval = static_cast<double>(val);
          h_view.pca()[j].candidateDirection() = Eigen::Vector3d(dval, dval + 0.1, dval + 0.2);
        }
      }

      // for the position multi view we restrict the iteration range for both views
      std::vector<int> offsetsPositionMultiView(nCollections);
      for (int i = 0; i < nCollections; ++i) {
        offsetsPositionMultiView[i] =
            sizesPositionLayout[i] / 3;  // restrict the iteration range of the position multi view
      }

      SoAPositionMultiView hostPositionMultiView(
          hostCollections,
          [](auto const& collection) { return collection.const_view().position(); },
          offsetsPositionMultiView);

      std::vector<PortableCollection<Device, SoA>> deviceCollections;

      for (int i = 0; i < nCollections; ++i) {
        deviceCollections.emplace_back(queue, sizesPositionLayout[i], sizesPCALayout[i]);
        alpaka::memcpy(queue, deviceCollections[i].buffer(), hostCollections[i].buffer());
      }

      SoAPositionMultiView positionMultiView(
          deviceCollections,
          [](auto const& collection) { return collection.const_view().position(); },
          offsetsPositionMultiView);
      SoAPCAMultiView pcaMultiView(deviceCollections,
                                   [](auto const& collection) { return collection.const_view().pca(); });

      REQUIRE(positionMultiView.size() ==
              std::accumulate(offsetsPositionMultiView.begin(), offsetsPositionMultiView.end(), 0));
      REQUIRE(pcaMultiView.size() == std::accumulate(sizesPCALayout.begin(), sizesPCALayout.end(), 0));
      REQUIRE(hostPositionMultiView.size() ==
              std::accumulate(offsetsPositionMultiView.begin(), offsetsPositionMultiView.end(), 0));

      REQUIRE(positionMultiView.numViews() == nCollections);
      REQUIRE(pcaMultiView.numViews() == nCollections);
      REQUIRE(hostPositionMultiView.numViews() == nCollections);

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
        int viewIdx = 0;
        for (int j = 0; j < nCollections; ++j) {
          if (i < pcaOffsets[j]) {
            viewIdx = j;
            break;
          }
        }
        auto h_view = hostCollections[viewIdx].const_view().pca();

        auto si = viewIdx == 0 ? h_view[i] : h_view[i - pcaOffsets[viewIdx - 1]];
        const float expected = si.vector_1() * si.vector_1() + si.vector_2() * si.vector_2() +
                               si.vector_3() * si.vector_3() +
                               static_cast<float>(si.candidateDirection().squaredNorm());
        REQUIRE(resultPCA_h[i] == Catch::Approx(expected).margin(1e-5));
      }
    }
  }
}
