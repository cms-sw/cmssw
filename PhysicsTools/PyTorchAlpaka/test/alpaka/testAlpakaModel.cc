#include <alpaka/alpaka.hpp>
#include <cppunit/extensions/HelperMacros.h>

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "PhysicsTools/PyTorchAlpaka/interface/GetDevice.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {

  constexpr auto modelPath = "PhysicsTools/PyTorchAlpaka/data/linear_dnn.pt";

  using namespace ALPAKA_ACCELERATOR_NAMESPACE::torch;

  class TestAlpakaModel : public CppUnit::TestFixture {
  public:
    void testCtorFromDevice();
    void testCtorFromQueue();
    void testMoveToDeviceFromAlpakaDevice();
    void testMoveToDeviceFromAlpakaQueue();
    void testAsyncExecution();

  private:
    CPPUNIT_TEST_SUITE(TestAlpakaModel);
    CPPUNIT_TEST(testCtorFromDevice);
    CPPUNIT_TEST(testCtorFromQueue);
    CPPUNIT_TEST(testMoveToDeviceFromAlpakaDevice);
    CPPUNIT_TEST(testMoveToDeviceFromAlpakaQueue);
    CPPUNIT_TEST(testAsyncExecution);
    CPPUNIT_TEST_SUITE_END();

    const int64_t batch_size_ = 8;

    template <typename Fn>
    void forEachAlpakaDevice(Fn&& fn) {
      auto m_path = edm::FileInPath(modelPath).fullPath();
      const auto& devices = cms::alpakatools::devices<Platform>();
      CPPUNIT_ASSERT(!devices.empty());
      for (auto& dev : devices) {
        std::cout << "Running test on device " << cms::torch::alpakatools::getDevice(dev) << std::endl;
        fn(dev, m_path);
      }
    }
  };

  CPPUNIT_TEST_SUITE_REGISTRATION(TestAlpakaModel);

  void TestAlpakaModel::testCtorFromDevice() {
    forEachAlpakaDevice([&](auto dev, auto m_path) {
      auto m = AlpakaModel(m_path, dev);
      CPPUNIT_ASSERT_EQUAL(cms::torch::alpakatools::getDevice(dev), m.device());
    });
  }

  void TestAlpakaModel::testCtorFromQueue() {
    forEachAlpakaDevice([&](auto dev, auto m_path) {
      Queue queue{dev};
      auto m = AlpakaModel(m_path, queue);
      CPPUNIT_ASSERT_EQUAL(cms::torch::alpakatools::getDevice(queue), m.device());
    });
  }

  void TestAlpakaModel::testMoveToDeviceFromAlpakaDevice() {
    forEachAlpakaDevice([&](auto dev, auto m_path) {
      auto m = AlpakaModel(m_path);
      m.to(dev);
      CPPUNIT_ASSERT_EQUAL(cms::torch::alpakatools::getDevice(dev), m.device());
    });
  }

  void TestAlpakaModel::testMoveToDeviceFromAlpakaQueue() {
    forEachAlpakaDevice([&](auto dev, auto m_path) {
      Queue queue{dev};
      auto m = AlpakaModel(m_path);
      m.to(queue);
      CPPUNIT_ASSERT_EQUAL(cms::torch::alpakatools::getDevice(queue), m.device());
    });
  }

  void TestAlpakaModel::testAsyncExecution() {
    // load model
    forEachAlpakaDevice([&](auto dev, auto m_path) {
      Queue queue{dev};
      auto m = AlpakaModel(m_path);

      // prepare input buffers
      auto inputs = std::vector<::torch::IValue>();
      inputs.push_back(::torch::randn({batch_size_, 3}, cms::torch::alpakatools::getDevice(queue)));

      // guard scope, restores when goes out of scope
      // all operations should be scheduled on provided queue.
      {
        // async model load and inference check
        m.to(queue);

        for (uint32_t i = 0; i < 10; ++i) {
          auto out = m.forward(inputs);
        }
      }

      // default queue should be restored and operations scheduled on default stream
      for (uint32_t i = 0; i < 10; ++i) {
        auto out = m.forward(inputs);
      }
    });
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest
