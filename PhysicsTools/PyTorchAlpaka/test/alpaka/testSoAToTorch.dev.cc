#include <cppunit/extensions/HelperMacros.h>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "PhysicsTools/PyTorch/test/testTorchBase.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/Config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {

  using namespace ALPAKA_ACCELERATOR_NAMESPACE::torch;
  using namespace cms::torch::alpakatools;

  // Input SOA
  GENERATE_SOA_LAYOUT(SoAPositionTemplate, SOA_COLUMN(float, x), SOA_COLUMN(float, y), SOA_COLUMN(float, z))

  using SoAPosition = SoAPositionTemplate<>;
  using SoAPositionView = SoAPosition::View;
  using SoAPositionConstView = SoAPosition::ConstView;

  // Output SOA
  GENERATE_SOA_LAYOUT(SoAResultTemplate, SOA_COLUMN(float, x), SOA_COLUMN(float, y))

  using SoAResult = SoAResultTemplate<>;
  using SoAResultView = SoAResult::View;

  class testSOAToTorch : public ::torchtest::testTorchBase {
    CPPUNIT_TEST_SUITE(testSOAToTorch);
    CPPUNIT_TEST(test);
    CPPUNIT_TEST_SUITE_END();

  public:
    std::string script() const override;
    void test();
  };

  CPPUNIT_TEST_SUITE_REGISTRATION(testSOAToTorch);

  std::string testSOAToTorch::script() const { return "testExportLinearDnn.py"; }

  class FillKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, PortableCollection<SoAPosition, Device>::View view) const {
      float input[4][3] = {{1, 2, 1}, {2, 4, 3}, {3, 4, 1}, {2, 3, 2}};

      for (int32_t i : cms::alpakatools::uniform_elements(acc, view.metadata().size())) {
        view.x()[i] = input[i][0];
        view.y()[i] = input[i][1];
        view.z()[i] = input[i][2];
      }
    }
  };

  class TestVerifyKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, PortableCollection<SoAResult, Device>::View view) const {
      float result_check[4][2] = {{2.3, -0.5}, {6.6, 3.0}, {2.5, -4.9}, {4.4, 1.3}};
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, view.metadata().size())) {
        ALPAKA_ASSERT_ACC(view.x()[i] - result_check[i][0] < 1.0e-05);
        ALPAKA_ASSERT_ACC(view.x()[i] - result_check[i][0] > -1.0e-05);
        ALPAKA_ASSERT_ACC(view.y()[i] - result_check[i][1] < 1.0e-05);
        ALPAKA_ASSERT_ACC(view.y()[i] - result_check[i][1] > -1.0e-05);
      }
    }
  };

  void fill(Queue& queue, PortableCollection<SoAPosition, Device>& collection) {
    uint32_t items = 64;
    uint32_t groups = cms::alpakatools::divide_up_by(collection->metadata().size(), items);
    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(groups, items);
    alpaka::exec<Acc1D>(queue, workDiv, FillKernel{}, collection.view());
  }

  void check(Queue& queue, PortableCollection<SoAResult, Device>& collection) {
    uint32_t items = 64;
    uint32_t groups = cms::alpakatools::divide_up_by(collection->metadata().size(), items);
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
    PortableCollection<SoAPosition, Device> positionCollection(batch_size, alpakaDevice);
    PortableCollection<SoAResult, Device> resultCollection(batch_size, alpakaDevice);
    fill(queue, positionCollection);

    // Deserialize the ScriptModule
    std::string model_path = modelPath() + "/linear_dnn.pt";
    auto model = AlpakaModel(model_path);
    model.to(queue);

    // Create SoA Metadata
    SoAMetadata input(batch_size);
    auto posview = positionCollection.view().records();
    input.append_block<SoAPosition>("main", posview.x(), posview.y(), posview.z());

    SoAMetadata output(batch_size);
    auto view = resultCollection.view().records();
    output.append_block<SoAResult>("result", view.x(), view.y());
    ModelMetadata metadata(input, output);

    // Call inference
    model.forward(queue, metadata);
    check(queue, resultCollection);

    PortableHostCollection<SoAResult> resultHostCollection(batch_size, cms::alpakatools::host());
    alpaka::memcpy(queue, resultHostCollection.buffer(), resultCollection.buffer());
    alpaka::wait(queue);

    for (uint32_t i = 0; i < batch_size; i++) {
      std::cout << "(" << resultHostCollection.view().x()[i] << ", " << resultHostCollection.view().y()[i] << ")"
                << std::endl;
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest