#include <alpaka/alpaka.hpp>
#include <cppunit/extensions/HelperMacros.h>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "PhysicsTools/PyTorch/test/testTorchBase.h"
#include "PhysicsTools/PyTorchAlpaka/interface/NvtxRAII.h"
#include "PhysicsTools/PyTorchAlpaka/interface/QueueGuard.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/Config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {

  using namespace ALPAKA_ACCELERATOR_NAMESPACE::torch;
  using namespace cms::torch::alpakatools;

  class TestAlpakaModel : public ::torchtest::testTorchBase {
  public:
    std::string script() const override;
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

    const int64_t batch_size_ = 2 << 10;
  };

  CPPUNIT_TEST_SUITE_REGISTRATION(TestAlpakaModel);

  std::string TestAlpakaModel::script() const { return "testExportLinearDnn.py"; }

  void TestAlpakaModel::testCtorFromDevice() {
    const auto& devices = cms::alpakatools::devices<Platform>();
    CPPUNIT_ASSERT(!devices.empty());
    const auto& dev = devices[0];

    auto m_path = modelPath() + "/linear_dnn.pt";
    auto m = AlpakaModel(m_path, dev);

    CPPUNIT_ASSERT_EQUAL(getDevice(dev), m.device());
  }

  void TestAlpakaModel::testCtorFromQueue() {
    const auto& devices = cms::alpakatools::devices<Platform>();
    CPPUNIT_ASSERT(!devices.empty());
    const auto& dev = devices[0];
    Queue queue{dev};

    auto m_path = modelPath() + "/linear_dnn.pt";
    auto m = AlpakaModel(m_path, queue);

    CPPUNIT_ASSERT_EQUAL(getDevice(queue), m.device());
  }

  void TestAlpakaModel::testMoveToDeviceFromAlpakaDevice() {
    const auto& devices = cms::alpakatools::devices<Platform>();
    CPPUNIT_ASSERT(!devices.empty());
    const auto& dev = devices[0];

    auto m_path = modelPath() + "/linear_dnn.pt";
    auto m = AlpakaModel(m_path);
    m.to(dev);

    CPPUNIT_ASSERT_EQUAL(getDevice(dev), m.device());
  }

  void TestAlpakaModel::testMoveToDeviceFromAlpakaQueue() {
    const auto& devices = cms::alpakatools::devices<Platform>();
    CPPUNIT_ASSERT(!devices.empty());
    const auto& dev = devices[0];
    Queue queue{dev};

    auto m_path = modelPath() + "/linear_dnn.pt";
    auto m = AlpakaModel(m_path);
    m.to(queue);

    CPPUNIT_ASSERT_EQUAL(getDevice(queue), m.device());
  }

  void TestAlpakaModel::testAsyncExecution() {
    NvtxRAII test_range("testAsyncExecution");

    // setup alpaka queue
    const auto& devices = cms::alpakatools::devices<Platform>();
    CPPUNIT_ASSERT(!devices.empty());
    const auto& dev = devices[0];
    Queue queue{dev};

    // load model
    auto m_path = modelPath() + "/linear_dnn.pt";
    auto m = AlpakaModel(m_path);

    // prepare input buffers
    NvtxRAII inbuf("inputBuffers");
    auto inputs = std::vector<::torch::IValue>();
    inputs.push_back(::torch::randn({batch_size_, 3}, getDevice(queue)));
    inbuf.end();

    // guard scope, restores when goes out of scope
    // all operations should be scheduled on provided queue.
    {
      QueueGuard<Queue> guard(queue);
      // async model load and inference check
      NvtxRAII exec1("execInExternalStream");
      NvtxRAII mmove("modelMoveToDevice");
      m.to(queue);
      mmove.end();

      for (uint32_t i = 0; i < 10; ++i) {
        NvtxRAII iter(("forwardPass:" + std::to_string(i)).c_str());
        auto out = m.forward(inputs);
        iter.end();
      }
      exec1.end();
    }

    // default queue should be restored and operations scheduled on default stream
    NvtxRAII exec2("execInDefaultStream");
    for (uint32_t i = 0; i < 10; ++i) {
      NvtxRAII iter(("forwardPass:" + std::to_string(i)).c_str());
      auto out = m.forward(inputs);
      iter.end();
    }
    exec2.end();
    test_range.end();
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest