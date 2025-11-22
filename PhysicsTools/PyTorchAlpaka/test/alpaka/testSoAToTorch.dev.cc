#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {

  constexpr auto modelPath = "PhysicsTools/PyTorchAlpaka/data/linear_dnn.pt";

  using namespace ALPAKA_ACCELERATOR_NAMESPACE::torch;

  // Input SOA
  GENERATE_SOA_LAYOUT(SoAPositionTemplate, SOA_COLUMN(float, x), SOA_COLUMN(float, y), SOA_COLUMN(float, z))

  using SoAPosition = SoAPositionTemplate<>;
  using PositionDeviceCollection = PortableCollection<SoAPosition, Device>;
  using PositionDeviceCollectionView = PortableCollection<SoAPosition, Device>::View;

  // Output SOA
  GENERATE_SOA_LAYOUT(SoAResultTemplate, SOA_COLUMN(float, x), SOA_COLUMN(float, y))

  using SoAResult = SoAResultTemplate<>;
  using ResultDeviceCollection = PortableCollection<SoAResult, Device>;
  using ResultDeviceCollectionView = PortableCollection<SoAResult, Device>::View;

  class testSOAToTorch : public CppUnit::TestFixture {
    CPPUNIT_TEST_SUITE(testSOAToTorch);
    CPPUNIT_TEST(test);
    CPPUNIT_TEST_SUITE_END();

  public:
    void test();
  };

  CPPUNIT_TEST_SUITE_REGISTRATION(testSOAToTorch);

  class FillKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, PositionDeviceCollectionView view) const {
      float input[4][3] = {{1.f, 2.f, 1.f}, {2.f, 4.f, 3.f}, {3.f, 4.f, 1.f}, {2.f, 3.f, 2.f}};
      for (auto i : cms::alpakatools::uniform_elements(acc, view.metadata().size())) {
        view.x()[i] = input[i][0];
        view.y()[i] = input[i][1];
        view.z()[i] = input[i][2];
      }
    }
  };

  class TestVerifyKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, ResultDeviceCollectionView view) const {
      float result_check[4][2] = {{2.3f, -0.5f}, {6.6f, 3.0f}, {2.5f, -4.9f}, {4.4f, 1.3f}};
      for (auto i : cms::alpakatools::uniform_elements(acc, view.metadata().size())) {
        ALPAKA_ASSERT_ACC(view.x()[i] - result_check[i][0] < 1.0e-05);
        ALPAKA_ASSERT_ACC(view.x()[i] - result_check[i][0] > -1.0e-05);
        ALPAKA_ASSERT_ACC(view.y()[i] - result_check[i][1] < 1.0e-05);
        ALPAKA_ASSERT_ACC(view.y()[i] - result_check[i][1] > -1.0e-05);
      }
    }
  };

  void fill(Queue& queue, PositionDeviceCollection& collection) {
    uint32_t items = 64;
    auto groups = cms::alpakatools::divide_up_by(collection->metadata().size(), items);
    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(groups, items);
    alpaka::exec<Acc1D>(queue, workDiv, FillKernel{}, collection.view());
  }

  void check(Queue& queue, ResultDeviceCollection& collection) {
    uint32_t items = 64;
    auto groups = cms::alpakatools::divide_up_by(collection->metadata().size(), items);
    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(groups, items);
    alpaka::exec<Acc1D>(queue, workDiv, TestVerifyKernel{}, collection.view());
  }

  void testSOAToTorch::test() {
    Platform platform;
    std::vector<Device> alpakaDevices = ::alpaka::getDevs(platform);
    const auto& alpakaHost = ::alpaka::getDevByIdx(alpaka_common::PlatformHost(), 0u);
    CPPUNIT_ASSERT(alpakaDevices.size());
    const auto& alpakaDevice = alpakaDevices[0];
    Queue queue{alpakaDevice};

    // Number of elements
    const std::size_t batch_size = 4;

    // Create and fill needed portable collections
    PositionDeviceCollection positionCollection(batch_size, alpakaDevice);
    ResultDeviceCollection resultCollection(batch_size, alpakaDevice);
    fill(queue, positionCollection);

    // Deserialize the ScriptModule
    std::string model_path = edm::FileInPath(modelPath).fullPath();
    auto model = AlpakaModel(model_path);
    model.to(queue);

    // Create SoA Metadata
    cms::torch::alpakatools::TensorCollection<Queue> input(batch_size);
    auto posRecords = positionCollection.const_view().records();
    input.add<SoAPosition>("main", posRecords.x(), posRecords.y(), posRecords.z());

    cms::torch::alpakatools::TensorCollection<Queue> output(batch_size);
    auto resultRecords = resultCollection.view().records();
    output.add<SoAResult>("result", resultRecords.x(), resultRecords.y());

    // Call inference
    model.forward(queue, input, output);
    check(queue, resultCollection);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest
