#include <alpaka/alpaka.hpp>
#include <cppunit/extensions/HelperMacros.h>
#include <sys/prctl.h>

#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorch/interface/AlpakaConfig.h"
#include "PhysicsTools/PyTorch/interface/Model.h"
#include "PhysicsTools/PyTorch/interface/Nvtx.h"
#include "PhysicsTools/PyTorch/test/testUtilities.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  GENERATE_SOA_LAYOUT(SoAInputsTemplate, SOA_COLUMN(float, x), SOA_COLUMN(float, y), SOA_COLUMN(float, z))

  GENERATE_SOA_LAYOUT(SoAOutputsTemplate, SOA_COLUMN(float, m), SOA_COLUMN(float, n))

  using SoAInputs = SoAInputsTemplate<>;
  using SoAOutputs = SoAOutputsTemplate<>;

  class testTorchExecutionControlGuard : public CppUnit::TestFixture {
    CPPUNIT_TEST_SUITE(testTorchExecutionControlGuard);
    CPPUNIT_TEST(test);
    CPPUNIT_TEST_SUITE_END();

  public:
    void test();
  };

  CPPUNIT_TEST_SUITE_REGISTRATION(testTorchExecutionControlGuard);

  void testTorchExecutionControlGuard::test() {
    if (prctl(PR_SET_NAME, "test::Main", 0, 0, 0))
      printf("Warning: Could not set thread name: %s\n", strerror(errno));

    // alpaka setup
    Platform platform;
    const auto& devices = alpaka::getDevs(platform);
    assert(!devices.empty());
    const auto& device = devices[0];

    uint32_t batch_size = 32;

    // host structs
    PortableHostCollection<SoAInputs> inputs_host(batch_size, cms::alpakatools::host());
    // prepare inputs
    for (size_t i = 0; i < batch_size; i++) {
      inputs_host.view().x()[i] = 0.0f;
      inputs_host.view().y()[i] = 0.0f;
      inputs_host.view().z()[i] = 0.0f;
    }

    cms::torch::alpaka::set_threading_guard();

    size_t num_threads = 8;
    std::vector<std::thread> threads;
    for (size_t t = 1; t <= num_threads; ++t) {
      threads.emplace_back([&, t] {
        std::cout << "T " << t << " Starting" << std::endl;
        Queue queue{device};
        cms::torch::alpaka::Guard<Queue> guard(queue);

        char threadName[15];
        snprintf(threadName, 15, "test::%ld", t);
        if (prctl(PR_SET_NAME, threadName, 0, 0, 0))
          printf("Warning: Could not set thread name: %s\n", strerror(errno));

        for (size_t i = 0; i < 10; i++) {
          NvtxScopedRange malloc_range((std::string("Malloc thread ") + std::to_string(t)).c_str());
          // host structs
          PortableHostCollection<SoAOutputs> outputs_host(batch_size, cms::alpakatools::host());
          // device structs
          std::cout << "T" << t << " I" << i << std::endl;
          PortableCollection<SoAInputs, Device> inputs_device(batch_size, queue);
          PortableCollection<SoAOutputs, Device> outputs_device(batch_size, queue);
          alpaka::memcpy(queue, inputs_device.buffer(), inputs_host.buffer());
          malloc_range.end();

          NvtxScopedRange minit_range((std::string("Model instantiation thread ") + std::to_string(t)).c_str());
          auto m_path = get_path("/src/PhysicsTools/PyTorch/models/jit_classification_model.pt");
          auto jit_model = cms::torch::alpaka::Model<cms::torch::alpaka::CompilationType::kJustInTime>(m_path);
          jit_model.to(queue);
          CPPUNIT_ASSERT(cms::torch::alpaka::device(queue) == jit_model.device());
          minit_range.end();

          NvtxScopedRange meta_range((std::string("Metarecords thread ") + std::to_string(t)).c_str());
          auto input_records = inputs_device.view().records();
          auto output_records = outputs_device.view().records();
          cms::torch::alpaka::SoAMetadata<SoAInputs> inputs_metadata(batch_size);
          inputs_metadata.append_block("features", input_records.x(), input_records.y(), input_records.z());
          cms::torch::alpaka::SoAMetadata<SoAOutputs> outputs_metadata(batch_size);
          outputs_metadata.append_block("preds", output_records.m(), output_records.n());
          cms::torch::alpaka::ModelMetadata<SoAInputs, SoAOutputs> metadata(inputs_metadata, outputs_metadata);
          meta_range.end();
          // inference
          NvtxScopedRange infer_range((std::string("Inference thread ") + std::to_string(t)).c_str());
          jit_model.forward(metadata);
          infer_range.end();
          // check outputs
          NvtxScopedRange assert_range((std::string("Assert thread ") + std::to_string(t)).c_str());
          alpaka::memcpy(queue, outputs_host.buffer(), outputs_device.buffer());
          alpaka::wait(queue);

          for (size_t i = 0; i < batch_size; i++) {
            CPPUNIT_ASSERT(outputs_host.const_view().m()[i] == 0.5f);
            CPPUNIT_ASSERT(outputs_host.const_view().n()[i] == 0.5f);
          }
          assert_range.end();
        }

        std::cout << "T " << t << " OK." << std::endl;
      });
    }

    for (auto& t : threads)
      t.join();
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
