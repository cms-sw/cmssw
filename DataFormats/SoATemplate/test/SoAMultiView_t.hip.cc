#include <Eigen/Core>
#include <Eigen/Dense>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/SoATemplate/interface/SoAMultiView.h"

#include <hip/hip_runtime.h>
#include "HeterogeneousCore/ROCmUtilities/interface/hipCheck.h"
#include "HeterogeneousCore/ROCmUtilities/interface/requireDevices.h"

GENERATE_SOA_LAYOUT(SoAPositionTemplate,
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z),
                    SOA_SCALAR(int, s1),
                    SOA_SCALAR(float, s2))

using SoAPosition = SoAPositionTemplate<>;
using SoAPositionView = SoAPosition::View;
using SoAPositionConstView = SoAPosition::ConstView;
using SoAPositionMultiView = SoAMultiView<SoAPositionConstView, 5>;

GENERATE_SOA_LAYOUT(SoAPCATemplate,
                    SOA_COLUMN(float, vector_1),
                    SOA_COLUMN(float, vector_2),
                    SOA_COLUMN(float, vector_3),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, candidateDirection))

using SoAPCA = SoAPCATemplate<>;
using SoAPCAView = SoAPCA::View;
using SoAPCAConstView = SoAPCA::ConstView;
using SoAPCAMultiView = SoAMultiView<SoAPCAConstView, 5>;

GENERATE_SOA_BLOCKS(SoABlocksTemplate, SOA_BLOCK(position, SoAPositionTemplate), SOA_BLOCK(pca, SoAPCATemplate))

using SoA = SoABlocksTemplate<>;
using SoAView = SoA::View;
using SoAConstView = SoA::ConstView;

__global__ void checkPositionMultiView(SoAPositionMultiView view, float* output) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= view.size())
    return;

  // For s1 we take the sum of all s1 values in the view, for s2 we take the value from the first view
  int s1 = 0;
  for (int j = 0; j < view.numViews(); ++j) {
    s1 += view.view(j).s1();
  }
  const float s2 = view.view(0).s2();

  auto si = view[i];
  output[i] = si.x() * si.x() + si.y() * si.y() + si.z() * si.z() + static_cast<float>(s1) + s2;
}

__global__ void checkPCAMultiView(SoAPCAMultiView view, float* output) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= view.size())
    return;
  auto si = view[i];
  output[i] = si.vector_1() * si.vector_1() + si.vector_2() * si.vector_2() + si.vector_3() * si.vector_3() +
              static_cast<float>(si.candidateDirection().squaredNorm());
}

TEST_CASE("SoAMultiView") {
  std::array<cms::soa::size_type, 2> sizes1{{17, 23}};
  // buffer size
  const cms::soa::size_type bufferSize1 = SoA::computeDataSize(sizes1);

  std::byte* h_buf1 = nullptr;
  hipCheck(hipHostMalloc(&h_buf1, bufferSize1));
  SoA h_soaLayout1(h_buf1, sizes1);
  SoAView h_view1(h_soaLayout1);

  // fill up
  for (cms::soa::size_type i = 0; i < sizes1[0]; i++) {
    h_view1.position()[i].x() = static_cast<float>(i);
    h_view1.position()[i].y() = static_cast<float>(i) * 2.0f;
    h_view1.position()[i].z() = static_cast<float>(i) * 3.0f;
  }
  h_view1.position().s1() = 42;
  h_view1.position().s2() = 42.43;
  for (cms::soa::size_type i = 0; i < sizes1[1]; i++) {
    h_view1.pca()[i].vector_1() = static_cast<float>(i);
    h_view1.pca()[i].vector_2() = static_cast<float>(i) * 2.0f;
    h_view1.pca()[i].vector_3() = static_cast<float>(i) * 3.0f;
    h_view1.pca()[i].candidateDirection() = Eigen::Vector3d(i, i * 2.0, i * 3.0);
  }

  std::array<cms::soa::size_type, 2> sizes2{{11, 17}};

  const cms::soa::size_type bufferSize2 = SoA::computeDataSize(sizes2);
  std::byte* h_buf2 = nullptr;
  hipCheck(hipHostMalloc(&h_buf2, bufferSize2));
  SoA h_soaLayout2(h_buf2, sizes2);
  SoAView h_view2(h_soaLayout2);

  // fill up
  for (cms::soa::size_type i = 0; i < sizes2[0]; i++) {
    h_view2.position()[i].x() = static_cast<float>(i) * 10.0f;
    h_view2.position()[i].y() = static_cast<float>(i) * 11.0f;
    h_view2.position()[i].z() = static_cast<float>(i) * 12.0f;
  }
  h_view2.position().s1() = 42;
  h_view2.position().s2() = 666.666;
  for (cms::soa::size_type i = 0; i < sizes2[1]; i++) {
    h_view2.pca()[i].vector_1() = static_cast<float>(i) * 17.0f;
    h_view2.pca()[i].vector_2() = static_cast<float>(i) * 18.0f;
    h_view2.pca()[i].vector_3() = static_cast<float>(i) * 19.0f;
    h_view2.pca()[i].candidateDirection() = Eigen::Vector3d(i * 111.0, i * 222.0, i * 333.0);
  }

  std::byte* d_buf1 = nullptr;
  hipCheck(hipMalloc(&d_buf1, bufferSize1));
  SoA d_soahdLayout1(d_buf1, sizes1);
  SoAConstView d_Constview(d_soahdLayout1);

  std::byte* d_buf2 = nullptr;
  hipCheck(hipMalloc(&d_buf2, bufferSize2));
  SoA d_soahdLayout2(d_buf2, sizes2);
  SoAConstView d_Constview2(d_soahdLayout2);

  hipCheck(hipMemcpy(d_buf1, h_buf1, bufferSize1, hipMemcpyHostToDevice));
  hipCheck(hipMemcpy(d_buf2, h_buf2, bufferSize2, hipMemcpyHostToDevice));

  std::vector<SoA> deviceSoAs;
  deviceSoAs.push_back(d_soahdLayout1);
  deviceSoAs.push_back(d_soahdLayout2);

  // for the position multi view we restrict the iteration range for both views
  std::vector<int> usedSizesForMultiview(5, 7);
  SoAPositionMultiView positionMultiView(
      deviceSoAs, [](SoA layout) -> auto { return SoAPositionConstView(layout.position()); }, usedSizesForMultiview);
  SoAPCAMultiView pcaMultiView(deviceSoAs, [](SoA layout) -> auto { return SoAPCAConstView(layout.pca()); });

  REQUIRE(positionMultiView.size() == usedSizesForMultiview[0] + usedSizesForMultiview[1]);
  REQUIRE(pcaMultiView.size() == sizes1[1] + sizes2[1]);

  REQUIRE(positionMultiView.numViews() == 2);
  REQUIRE(pcaMultiView.numViews() == 2);

  float* d_outputPosition = nullptr;
  const cms::soa::size_type outputSizePosition = positionMultiView.size() * sizeof(float);
  hipCheck(hipMalloc(&d_outputPosition, outputSizePosition));
  checkPositionMultiView<<<(positionMultiView.size() + 255) / 256, 256>>>(positionMultiView, d_outputPosition);
  hipCheck(hipDeviceSynchronize());
  std::vector<float> h_outputPosition(positionMultiView.size());
  hipCheck(hipMemcpy(h_outputPosition.data(), d_outputPosition, outputSizePosition, hipMemcpyDeviceToHost));

  // check results
  for (cms::soa::size_type i = 0; i < positionMultiView.size(); ++i) {
    int s1 = 0;
    for (int j = 0; j < positionMultiView.numViews(); ++j) {
      s1 += positionMultiView.view(j).s1();
    }
    auto si = i < usedSizesForMultiview[0] ? h_view1.position()[i] : h_view2.position()[i - usedSizesForMultiview[0]];
    const float expected =
        si.x() * si.x() + si.y() * si.y() + si.z() * si.z() + static_cast<float>(s1) + h_view1.position().s2();
    REQUIRE(h_outputPosition[i] == Catch::Approx(expected).margin(1e-5));
  }

  float* d_outputPCA = nullptr;
  const cms::soa::size_type outputSizePCA = pcaMultiView.size() * sizeof(float);
  hipCheck(hipMalloc(&d_outputPCA, outputSizePCA));
  checkPCAMultiView<<<(pcaMultiView.size() + 255) / 256, 256>>>(pcaMultiView, d_outputPCA);
  hipCheck(hipDeviceSynchronize());
  std::vector<float> h_outputPCA(pcaMultiView.size());
  hipCheck(hipMemcpy(h_outputPCA.data(), d_outputPCA, outputSizePCA, hipMemcpyDeviceToHost));

  // check results
  for (cms::soa::size_type i = 0; i < pcaMultiView.size(); ++i) {
    auto si = i < sizes1[1] ? h_view1.pca()[i] : h_view2.pca()[i - sizes1[1]];
    const float expected = si.vector_1() * si.vector_1() + si.vector_2() * si.vector_2() +
                           si.vector_3() * si.vector_3() + static_cast<float>(si.candidateDirection().squaredNorm());
    REQUIRE(h_outputPCA[i] == Catch::Approx(expected).margin(1e-5));
  }

  hipCheck(hipFreeHost(h_buf1));
  hipCheck(hipFreeHost(h_buf2));
  hipCheck(hipFree(d_buf1));
  hipCheck(hipFree(d_buf2));
  hipCheck(hipFree(d_outputPosition));
}
