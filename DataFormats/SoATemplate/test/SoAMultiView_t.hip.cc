#include <algorithm>

#include <Eigen/Core>
#include <Eigen/Dense>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/SoATemplate/interface/SoAConstMultiView.h"

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
using SoAPositionMultiView = SoAConstMultiView<SoAPositionConstView, 5>;

using AoSPosition = SoAPositionTemplate<>::AoSWrapper;
using AoSPositionView = AoSPosition::View;
using AoSPositionConstView = AoSPosition::ConstView;
using AoSPositionMultiView = SoAConstMultiView<AoSPosition::ConstView, 5>;

GENERATE_SOA_LAYOUT(SoAPCATemplate,
                    SOA_COLUMN(float, vector_1),
                    SOA_COLUMN(float, vector_2),
                    SOA_COLUMN(float, vector_3),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, candidateDirection))

using SoAPCA = SoAPCATemplate<>;
using SoAPCAView = SoAPCA::View;
using SoAPCAConstView = SoAPCA::ConstView;
using SoAPCAMultiView = SoAConstMultiView<SoAPCAConstView, 5>;

using AoSPCA = SoAPCATemplate<>::AoSWrapper;
using AoSPCAView = AoSPCA::View;
using AoSPCAConstView = AoSPCA::ConstView;
using AoSPCAMultiView = SoAConstMultiView<AoSPCAConstView, 5>;

GENERATE_SOA_BLOCKS(SoABlocksTemplate, SOA_BLOCK(position, SoAPositionTemplate), SOA_BLOCK(pca, SoAPCATemplate))

using SoA = SoABlocksTemplate<>;
using SoAView = SoA::View;
using SoAConstView = SoA::ConstView;

using AoS = SoABlocksTemplate<>::AoSWrapper;
using AoSView = AoS::View;
using AoSConstView = AoS::ConstView;

__global__ void transpose(AoSView dst, SoAConstView src) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  dst.transpose(src, i);
}

template <typename PositionMultiView>
__global__ void checkPositionMultiView(PositionMultiView view, float* output) {
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

template <typename PCAMultiView>
__global__ void checkPCAMultiView(PCAMultiView view, float* output) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= view.size())
    return;
  auto si = view[i];
  output[i] = si.vector_1() * si.vector_1() + si.vector_2() * si.vector_2() + si.vector_3() * si.vector_3() +
              static_cast<float>(si.candidateDirection().squaredNorm());
}

template <typename PositionMultiView>
void checkPositionMultiViewResult(const PositionMultiView positionMultiView,
                                  const SoAPositionMultiView hostPositionMultiView) {
  float* d_outputPosition = nullptr;
  const cms::soa::size_type outputSizePosition = positionMultiView.size() * sizeof(float);
  HIP_CHECK(hipHostMalloc(&d_outputPosition, outputSizePosition));
  checkPositionMultiView<<<(positionMultiView.size() + 255) / 256, 256>>>(positionMultiView, d_outputPosition);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> h_outputPosition(positionMultiView.size());
  HIP_CHECK(hipMemcpy(h_outputPosition.data(), d_outputPosition, outputSizePosition, hipMemcpyDeviceToHost));

  // check results
  for (cms::soa::size_type i = 0; i < hostPositionMultiView.size(); ++i) {
    int s1 = 0;
    for (int j = 0; j < hostPositionMultiView.numViews(); ++j) {
      s1 += hostPositionMultiView.view(j).s1();
    }
    auto const s2 = hostPositionMultiView.view(0).s2();
    auto si = hostPositionMultiView[i];
    const float expected = si.x() * si.x() + si.y() * si.y() + si.z() * si.z() + static_cast<float>(s1) + s2;
    REQUIRE(h_outputPosition[i] == Catch::Approx(expected).margin(1e-5));
  }
  HIP_CHECK(hipFree(d_outputPosition));
}

template <typename PCAMultiView, typename PCAView>
void checkPCAMultiViewResult(const PCAMultiView pcaMultiView,
                             const PCAView h_view1,
                             const PCAView h_view2,
                             const cms::soa::size_type offset) {
  float* d_outputPCA = nullptr;
  const cms::soa::size_type outputSizePCA = pcaMultiView.size() * sizeof(float);
  HIP_CHECK(hipHostMalloc(&d_outputPCA, outputSizePCA));
  checkPCAMultiView<<<(pcaMultiView.size() + 255) / 256, 256>>>(pcaMultiView, d_outputPCA);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> h_outputPCA(pcaMultiView.size());
  HIP_CHECK(hipMemcpy(h_outputPCA.data(), d_outputPCA, outputSizePCA, hipMemcpyDeviceToHost));

  // check results
  for (cms::soa::size_type i = 0; i < pcaMultiView.size(); ++i) {
    auto si = i < offset ? h_view1.pca()[i] : h_view2.pca()[i - offset];
    const float expected = si.vector_1() * si.vector_1() + si.vector_2() * si.vector_2() +
                           si.vector_3() * si.vector_3() + static_cast<float>(si.candidateDirection().squaredNorm());
    REQUIRE(h_outputPCA[i] == Catch::Approx(expected).margin(1e-5));
  }
  HIP_CHECK(hipFree(d_outputPCA));
}

TEST_CASE("SoAConstMultiViewHIP") {
  std::array<cms::soa::size_type, 2> sizes1{{17, 23}};
  // buffer size
  const cms::soa::size_type bufferSize1 = SoA::computeDataSize(sizes1);

  std::byte* h_buf1 = nullptr;
  HIP_CHECK(hipHostMalloc(&h_buf1, bufferSize1));
  SoA h_soaLayout1(h_buf1, sizes1);
  SoAView h_view1(h_soaLayout1);
  SoAConstView h_constView1(h_soaLayout1);

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

  std::array<cms::soa::size_type, 2> sizes2{{11, 17}};

  const cms::soa::size_type bufferSize2 = SoA::computeDataSize(sizes2);
  std::byte* h_buf2 = nullptr;
  HIP_CHECK(hipHostMalloc(&h_buf2, bufferSize2));
  SoA h_soaLayout2(h_buf2, sizes2);
  SoAView h_view2(h_soaLayout2);
  SoAConstView h_constView2(h_soaLayout2);

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

  // for the position multi view we restrict the iteration range for both views
  std::vector<int> usedSizesForMultiview{5, 7};
  std::vector<SoA> hostSoAs;
  hostSoAs.push_back(h_soaLayout1);
  hostSoAs.push_back(h_soaLayout2);
  SoAPositionMultiView hostPositionMultiView(
      hostSoAs, [](SoA layout) -> auto { return SoAPositionConstView(layout.position()); }, usedSizesForMultiview);

  std::byte* d_buf1 = nullptr;
  HIP_CHECK(hipMalloc(&d_buf1, bufferSize1));
  SoA d_soahdLayout1(d_buf1, sizes1);
  SoAConstView d_Constview(d_soahdLayout1);

  std::byte* d_buf2 = nullptr;
  HIP_CHECK(hipMalloc(&d_buf2, bufferSize2));
  SoA d_soahdLayout2(d_buf2, sizes2);
  SoAConstView d_Constview2(d_soahdLayout2);

  HIP_CHECK(hipMemcpy(d_buf1, h_buf1, bufferSize1, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_buf2, h_buf2, bufferSize2, hipMemcpyHostToDevice));

  SECTION("Check SoA MultiView") {
    std::vector<SoA> deviceSoAs;
    deviceSoAs.push_back(d_soahdLayout1);
    deviceSoAs.push_back(d_soahdLayout2);

    SoAPositionMultiView positionMultiView(
        deviceSoAs, [](SoA layout) -> auto { return SoAPositionConstView(layout.position()); }, usedSizesForMultiview);
    SoAPCAMultiView pcaMultiView(deviceSoAs, [](SoA layout) -> auto { return SoAPCAConstView(layout.pca()); });

    REQUIRE(positionMultiView.size() == usedSizesForMultiview[0] + usedSizesForMultiview[1]);
    REQUIRE(pcaMultiView.size() == sizes1[1] + sizes2[1]);
    REQUIRE(hostPositionMultiView.size() == usedSizesForMultiview[0] + usedSizesForMultiview[1]);

    REQUIRE(positionMultiView.numViews() == 2);
    REQUIRE(pcaMultiView.numViews() == 2);
    REQUIRE(hostPositionMultiView.numViews() == 2);

    checkPositionMultiViewResult(positionMultiView, hostPositionMultiView);
    checkPCAMultiViewResult(pcaMultiView, h_constView1, h_constView2, sizes1[1]);
  }

  SECTION("Check AoS MultiView") {
    std::byte* d_aos_buf1 = nullptr;
    HIP_CHECK(hipMalloc(&d_aos_buf1, AoS::computeDataSize(sizes1)));
    AoS d_aosLayout1(d_aos_buf1, sizes1);
    AoSView d_AoSView1(d_aosLayout1);
    AoSConstView d_AoSConstview1(d_aosLayout1);

    std::byte* d_aos_buf2 = nullptr;
    HIP_CHECK(hipMalloc(&d_aos_buf2, AoS::computeDataSize(sizes2)));
    AoS d_aosLayout2(d_aos_buf2, sizes2);
    AoSView d_AoSView2(d_aosLayout2);
    AoSConstView d_AoSConstview2(d_aosLayout2);

    const auto largestSize1 = *std::max_element(sizes1.begin(), sizes1.end());
    transpose<<<(largestSize1 + 255) / 256, 256>>>(d_AoSView1, d_Constview);

    const auto largestSize2 = *std::max_element(sizes2.begin(), sizes2.end());
    transpose<<<(largestSize2 + 255) / 256, 256>>>(d_AoSView2, d_Constview2);

    std::vector<AoS> deviceAoSs;
    deviceAoSs.push_back(d_aosLayout1);
    deviceAoSs.push_back(d_aosLayout2);

    // for the position multi view we restrict the iteration range for both views
    AoSPositionMultiView positionMultiView(
        deviceAoSs, [](AoS layout) -> auto { return AoSPositionConstView(layout.position()); }, usedSizesForMultiview);
    AoSPCAMultiView pcaMultiView(deviceAoSs, [](AoS layout) -> auto { return AoSPCAConstView(layout.pca()); });

    REQUIRE(positionMultiView.size() == usedSizesForMultiview[0] + usedSizesForMultiview[1]);
    REQUIRE(pcaMultiView.size() == sizes1[1] + sizes2[1]);

    REQUIRE(positionMultiView.numViews() == 2);
    REQUIRE(pcaMultiView.numViews() == 2);

    checkPositionMultiViewResult(positionMultiView, hostPositionMultiView);
    checkPCAMultiViewResult(pcaMultiView, h_constView1, h_constView2, sizes1[1]);

    HIP_CHECK(hipFree(d_aos_buf1));
    HIP_CHECK(hipFree(d_aos_buf2));
  }

  HIP_CHECK(hipFreeHost(h_buf1));
  HIP_CHECK(hipFreeHost(h_buf2));
  HIP_CHECK(hipFree(d_buf1));
  HIP_CHECK(hipFree(d_buf2));
}
