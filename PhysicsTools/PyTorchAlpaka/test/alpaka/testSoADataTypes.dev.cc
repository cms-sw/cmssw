#include <alpaka/alpaka.hpp>
#include <cppunit/extensions/HelperMacros.h>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "PhysicsTools/PyTorchAlpaka/interface/GetDevice.h"
#include "PhysicsTools/PyTorchAlpaka/interface/SoAConversion.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {

  class TestSOADataTypesAlpaka : public CppUnit::TestFixture {
    CPPUNIT_TEST_SUITE(TestSOADataTypesAlpaka);
    CPPUNIT_TEST(testInterface);
    CPPUNIT_TEST(testMultiOutput);
    CPPUNIT_TEST(testSingleElement);
    CPPUNIT_TEST(testNoElement);
    CPPUNIT_TEST(testEmptyMetadata);
    CPPUNIT_TEST_SUITE_END();

  public:
    void testInterface();
    void testMultiOutput();
    void testSingleElement();
    void testNoElement();
    void testEmptyMetadata();
  };

  CPPUNIT_TEST_SUITE_REGISTRATION(TestSOADataTypesAlpaka);

  GENERATE_SOA_LAYOUT(SoATemplate,
                      SOA_EIGEN_COLUMN(Eigen::Vector3d, a),
                      SOA_EIGEN_COLUMN(Eigen::Vector3d, b),

                      SOA_EIGEN_COLUMN(Eigen::Matrix2f, c),

                      SOA_COLUMN(double, x),
                      SOA_COLUMN(double, y),
                      SOA_COLUMN(double, z),

                      SOA_SCALAR(float, type),
                      SOA_SCALAR(int, someNumber),

                      SOA_COLUMN(double, v),
                      SOA_COLUMN(double, w));

  using SoA = SoATemplate<>;
  using SoAView = SoA::View;
  using SoACollection = PortableCollection<SoA, Device>;
  using SoACollectionView = PortableCollection<SoA, Device>::View;
  using SoAHostCollection = PortableHostCollection<SoA>;
  using SoAHostCollectionView = PortableHostCollection<SoA>::View;

  constexpr auto tol = 1.0e-5;

  class FillKernel {
  public:
    template <typename TAcc>
      requires ::alpaka::isAccelerator<TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, SoACollectionView view) const {
      if (cms::alpakatools::once_per_grid(acc)) {
        view.type() = 4;
        view.someNumber() = 5;
      }

      for (auto i : cms::alpakatools::uniform_elements(acc, view.metadata().size())) {
        view[i].a()(0) = 1 + i;
        view[i].a()(1) = 2 + i;
        view[i].a()(2) = 3 + i;

        view[i].b()(0) = 4 + i;
        view[i].b()(1) = 5 + i;
        view[i].b()(2) = 6 + i;

        view[i].c()(0, 0) = 4 + i;
        view[i].c()(0, 1) = 6 + i;
        view[i].c()(1, 0) = 8 + i;
        view[i].c()(1, 1) = 10 + i;

        view.x()[i] = 12 + i;
        view.y()[i] = 1 + 2.5 * i;
        view.z()[i] = 36 * i;
      }
    }
  };

  class InputVerifyKernel {
  public:
    template <typename TAcc>
      requires ::alpaka::isAccelerator<TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, SoACollectionView view) const {
      if (cms::alpakatools::once_per_grid(acc)) {
        ALPAKA_ASSERT_ACC(view.type() == 4);
        ALPAKA_ASSERT_ACC(view.someNumber() == 5);
      }

      for (auto i : cms::alpakatools::uniform_elements(acc, view.metadata().size())) {
        ALPAKA_ASSERT_ACC(view[i].a()(0) == 1 + i);
        ALPAKA_ASSERT_ACC(view[i].a()(1) == 2 + i);
        ALPAKA_ASSERT_ACC(view[i].a()(2) == 3 + i);

        ALPAKA_ASSERT_ACC(view[i].b()(0) == 4 + i);
        ALPAKA_ASSERT_ACC(view[i].b()(1) == 5 + i);
        ALPAKA_ASSERT_ACC(view[i].b()(2) == 6 + i);

        ALPAKA_ASSERT_ACC(view[i].c()(0, 0) == 4 + i);
        ALPAKA_ASSERT_ACC(view[i].c()(0, 1) == 6 + i);
        ALPAKA_ASSERT_ACC(view[i].c()(1, 0) == 8 + i);
        ALPAKA_ASSERT_ACC(view[i].c()(1, 1) == 10 + i);

        ALPAKA_ASSERT_ACC(view.x()[i] == 12 + i);
        ALPAKA_ASSERT_ACC(view.y()[i] == 1 + 2.5 * i);
        ALPAKA_ASSERT_ACC(view.z()[i] == 36 * i);
      }
    }
  };

  class TestOutputVerifyKernel {
  public:
    template <typename TAcc>
      requires ::alpaka::isAccelerator<TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, SoACollectionView view) const {
      for (auto i : cms::alpakatools::uniform_elements(acc, view.metadata().size())) {
        ALPAKA_ASSERT_ACC(view.x()[i] - view.v()[i] < tol);
        ALPAKA_ASSERT_ACC(view.x()[i] - view.v()[i] > -tol);

        ALPAKA_ASSERT_ACC(view.y()[i] - view.w()[i] < tol);
        ALPAKA_ASSERT_ACC(view.y()[i] - view.w()[i] > -tol);
      }
    }
  };

  void fill(Queue& queue, SoACollection& collection) {
    size_t items = 64;
    auto groups = cms::alpakatools::divide_up_by(collection->metadata().size(), items);
    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(groups, items);
    ::alpaka::exec<Acc1D>(queue, workDiv, FillKernel{}, collection.view());
    ::alpaka::exec<Acc1D>(queue, workDiv, InputVerifyKernel{}, collection.view());
  }

  void check_output(Queue& queue, SoACollection& collection) {
    size_t items = 64;
    auto groups = cms::alpakatools::divide_up_by(collection->metadata().size(), items);
    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(groups, items);
    ::alpaka::exec<Acc1D>(queue, workDiv, TestOutputVerifyKernel{}, collection.view());
  }

  void check(SoAHostCollectionView& view, std::vector<::torch::IValue>& tensors) {
    // Check if tensor list built correctly
    for (int i = 0; i < view.metadata().size(); i++) {
      CPPUNIT_ASSERT(view[i].a()(0) - tensors[3].toTensor()[i][0][0].item<double>() < tol);
      CPPUNIT_ASSERT(view[i].a()(1) - tensors[3].toTensor()[i][0][1].item<double>() < tol);
      CPPUNIT_ASSERT(view[i].a()(2) - tensors[3].toTensor()[i][0][2].item<double>() < tol);
      CPPUNIT_ASSERT(view[i].a()(0) - tensors[3].toTensor()[i][0][0].item<double>() > -tol);
      CPPUNIT_ASSERT(view[i].a()(1) - tensors[3].toTensor()[i][0][1].item<double>() > -tol);
      CPPUNIT_ASSERT(view[i].a()(2) - tensors[3].toTensor()[i][0][2].item<double>() > -tol);

      CPPUNIT_ASSERT(view[i].b()(0) - tensors[3].toTensor()[i][1][0].item<double>() < tol);
      CPPUNIT_ASSERT(view[i].b()(1) - tensors[3].toTensor()[i][1][1].item<double>() < tol);
      CPPUNIT_ASSERT(view[i].b()(2) - tensors[3].toTensor()[i][1][2].item<double>() < tol);
      CPPUNIT_ASSERT(view[i].b()(0) - tensors[3].toTensor()[i][1][0].item<double>() > -tol);
      CPPUNIT_ASSERT(view[i].b()(1) - tensors[3].toTensor()[i][1][1].item<double>() > -tol);
      CPPUNIT_ASSERT(view[i].b()(2) - tensors[3].toTensor()[i][1][2].item<double>() > -tol);

      CPPUNIT_ASSERT(view[i].c()(0, 0) - tensors[2].toTensor()[i][0][0].item<float>() < tol);
      CPPUNIT_ASSERT(view[i].c()(0, 0) - tensors[2].toTensor()[i][0][0].item<float>() > -tol);
      CPPUNIT_ASSERT(view[i].c()(0, 1) - tensors[2].toTensor()[i][0][1].item<float>() < tol);
      CPPUNIT_ASSERT(view[i].c()(0, 1) - tensors[2].toTensor()[i][0][1].item<float>() > -tol);
      CPPUNIT_ASSERT(view[i].c()(1, 0) - tensors[2].toTensor()[i][1][0].item<float>() < tol);
      CPPUNIT_ASSERT(view[i].c()(1, 0) - tensors[2].toTensor()[i][1][0].item<float>() > -tol);
      CPPUNIT_ASSERT(view[i].c()(1, 1) - tensors[2].toTensor()[i][1][1].item<float>() < tol);
      CPPUNIT_ASSERT(view[i].c()(1, 1) - tensors[2].toTensor()[i][1][1].item<float>() > -tol);

      CPPUNIT_ASSERT(view.x()[i] - tensors[0].toTensor()[i][0].item<double>() < tol);
      CPPUNIT_ASSERT(view.x()[i] - tensors[0].toTensor()[i][0].item<double>() > -tol);

      CPPUNIT_ASSERT(view.x()[i] - tensors[4].toTensor()[i].item<double>() < tol);
      CPPUNIT_ASSERT(view.x()[i] - tensors[4].toTensor()[i].item<double>() > -tol);

      CPPUNIT_ASSERT(view.y()[i] - tensors[0].toTensor()[i][1].item<double>() < tol);
      CPPUNIT_ASSERT(view.y()[i] - tensors[0].toTensor()[i][1].item<double>() > -tol);

      CPPUNIT_ASSERT(view.z()[i] - tensors[0].toTensor()[i][2].item<double>() < tol);
      CPPUNIT_ASSERT(view.z()[i] - tensors[0].toTensor()[i][2].item<double>() > -tol);

      CPPUNIT_ASSERT(view.type() - tensors[1].toTensor()[i].item<float>() < tol);
      CPPUNIT_ASSERT(view.type() - tensors[1].toTensor()[i].item<float>() > -tol);
    }
  }

  void TestSOADataTypesAlpaka::testInterface() {
    Platform platform;
    std::vector<Device> alpakaDevices = ::alpaka::getDevs(platform);
    const auto& alpakaHost = ::alpaka::getDevByIdx(::alpaka_common::PlatformHost(), 0u);
    CPPUNIT_ASSERT(alpakaDevices.size());
    const auto& alpakaDevice = alpakaDevices[0];
    Queue queue{alpakaDevice};
    ::torch::Device torchDevice = cms::torch::alpakatools::getDevice(queue);

    // Large batch size, so multiple bunches needed
    const std::size_t batch_size = 64;

    // Create and fill needed portable collections
    SoACollection deviceCollection(batch_size, queue);
    fill(queue, deviceCollection);
    auto records = deviceCollection.view().records();

    cms::torch::alpakatools::TensorCollection<Queue> input(batch_size);
    input.add<SoA>("vector", records.a(), records.b());
    input.add<SoA>("single_vector", records.a());
    input.add<SoA>("matrix", records.c());
    input.add<SoA>("column", records.x(), records.y(), records.z());
    input.add<SoA>("single_column", records.x());
    input.add<SoA>("scalar", records.type());
    input.change_order({"column", "scalar", "matrix", "vector", "single_column", "single_vector"});

    std::vector<::torch::IValue> tensors = cms::torch::alpakatools::detail::convertInput(input, torchDevice);

    SoAHostCollection hostCollection(batch_size, queue);
    alpaka::memcpy(queue, hostCollection.buffer(), deviceCollection.buffer());
    alpaka::wait(queue);

    check(hostCollection.view(), tensors);
  };

  void TestSOADataTypesAlpaka::testMultiOutput() {
    Platform platform;
    std::vector<Device> alpakaDevices = ::alpaka::getDevs(platform);
    const auto& alpakaHost = ::alpaka::getDevByIdx(::alpaka_common::PlatformHost(), 0u);
    CPPUNIT_ASSERT(alpakaDevices.size());
    const auto& alpakaDevice = alpakaDevices[0];
    Queue queue{alpakaDevice};
    ::torch::Device torchDevice = cms::torch::alpakatools::getDevice(queue);

    // Large batch size, so multiple bunches needed
    const std::size_t batch_size = 64;

    // Create and fill needed portable collections
    SoACollection deviceCollection(batch_size, queue);
    fill(queue, deviceCollection);

    auto records = deviceCollection.view().records();
    cms::torch::alpakatools::TensorCollection<Queue> input(batch_size);
    input.add<SoA>("x", records.x());
    input.add<SoA>("y", records.y());

    cms::torch::alpakatools::TensorCollection<Queue> output(batch_size);
    output.add<SoA>("v", records.v());
    output.add<SoA>("w", records.w());

    std::vector<::torch::IValue> tensors = cms::torch::alpakatools::detail::convertInput(input, torchDevice);
    cms::torch::alpakatools::detail::convertOutput(tensors, output, torchDevice);

    // Check if tensor list built correctly
    check_output(queue, deviceCollection);
  };

  void TestSOADataTypesAlpaka::testSingleElement() {
    Platform platform;
    std::vector<Device> alpakaDevices = ::alpaka::getDevs(platform);
    CPPUNIT_ASSERT(alpakaDevices.size());
    const auto& alpakaDevice = alpakaDevices[0];
    Queue queue(alpakaDevice);
    ::torch::Device torchDevice = cms::torch::alpakatools::getDevice(queue);

    // Create and fill portable collections
    const std::size_t batch_size = 1;
    SoACollection deviceCollection(batch_size, queue);
    fill(queue, deviceCollection);
    auto records = deviceCollection.view().records();

    // Run Converter for single tensor
    cms::torch::alpakatools::TensorCollection<Queue> input(batch_size);
    input.add<SoA>("vector", records.a(), records.b());
    input.add<SoA>("single_vector", records.a(), records.b());
    input.add<SoA>("matrix", records.c());
    input.add<SoA>("column", records.x(), records.y(), records.z());
    input.add<SoA>("single_column", records.x());
    input.add<SoA>("scalar", records.type());
    input.change_order({"column", "scalar", "matrix", "vector", "single_column", "single_vector"});

    cms::torch::alpakatools::TensorCollection<Queue> output(batch_size);
    output.add<SoA>("result", records.v());

    std::vector<::torch::IValue> tensors = cms::torch::alpakatools::detail::convertInput(input, torchDevice);

    // Check if tensor list built correctly
    SoAHostCollection hostCollection(batch_size, queue);
    ::alpaka::memcpy(queue, hostCollection.buffer(), deviceCollection.buffer());
    check(hostCollection.view(), tensors);
  };

  void TestSOADataTypesAlpaka::testNoElement() {
    Platform platform;
    std::vector<Device> alpakaDevices = ::alpaka::getDevs(platform);
    CPPUNIT_ASSERT(alpakaDevices.size());
    const auto& alpakaDevice = alpakaDevices[0];
    Queue queue(alpakaDevice);
    ::torch::Device torchDevice = cms::torch::alpakatools::getDevice(queue);

    //Create empty portable collection
    const std::size_t batch_size = 0;
    SoACollection deviceCollection(batch_size, queue);
    auto records = deviceCollection.view().records();

    // Run Converter
    cms::torch::alpakatools::TensorCollection<Queue> input(batch_size);
    input.add<SoA>("vector", records.a(), records.b());
    input.add<SoA>("matrix", records.c());
    input.add<SoA>("column", records.x(), records.y(), records.z());
    input.add<SoA>("scalar", records.type());
    input.change_order({"column", "scalar", "matrix", "vector"});

    std::vector<::torch::IValue> tensors = cms::torch::alpakatools::detail::convertInput(input, torchDevice);

    // Check if tensor list has empty tensors
    CPPUNIT_ASSERT(tensors[0].toTensor().size(0) == 0);
    CPPUNIT_ASSERT(tensors[1].toTensor().size(0) == 0);
    CPPUNIT_ASSERT(tensors[2].toTensor().size(0) == 0);
    CPPUNIT_ASSERT(tensors[3].toTensor().size(0) == 0);
  };

  void TestSOADataTypesAlpaka::testEmptyMetadata() {
    // alpaka setup
    Platform platform;
    std::vector<Device> alpakaDevices = ::alpaka::getDevs(platform);
    CPPUNIT_ASSERT(alpakaDevices.size());
    const auto& alpakaDevice = alpakaDevices[0];
    Queue queue(alpakaDevice);
    ::torch::Device torchDevice = cms::torch::alpakatools::getDevice(queue);

    // Create and fill portable collections
    const std::size_t batch_size = 32;
    SoACollection deviceCollection(batch_size, queue);
    fill(queue, deviceCollection);

    // Run Converter for empty metadata
    cms::torch::alpakatools::TensorCollection<Queue> input(batch_size);

    std::vector<::torch::IValue> tensors = cms::torch::alpakatools::detail::convertInput(input, torchDevice);

    // Check if tensor list is empty
    CPPUNIT_ASSERT(tensors.size() == 0);
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest
