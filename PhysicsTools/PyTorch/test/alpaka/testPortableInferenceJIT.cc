#include <alpaka/alpaka.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <cppunit/extensions/HelperMacros.h>

#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "PhysicsTools/PyTorch/interface/AlpakaConfig.h"
#include "PhysicsTools/PyTorch/interface/Model.h"
#include "PhysicsTools/PyTorch/interface/SoAMetadata.h"
#include "PhysicsTools/PyTorch/test/testUtilities.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  GENERATE_SOA_LAYOUT(SoAInputsTemplate, SOA_COLUMN(float, x), SOA_COLUMN(float, y), SOA_COLUMN(float, z))

  GENERATE_SOA_LAYOUT(SoAOutputsTemplate, SOA_COLUMN(float, m), SOA_COLUMN(float, n))

  using SoAInputs = SoAInputsTemplate<>;
  using SoAOutputs = SoAOutputsTemplate<>;

  class TestPortableInferenceJIT : public CppUnit::TestFixture {
    CPPUNIT_TEST_SUITE(TestPortableInferenceJIT);
    CPPUNIT_TEST(test);
    CPPUNIT_TEST_SUITE_END();

  public:
    void test();
  };

  CPPUNIT_TEST_SUITE_REGISTRATION(TestPortableInferenceJIT);

  void TestPortableInferenceJIT::test() {
    // alpaka setup
    Platform platform;
    std::vector<Device> alpaka_devices = alpaka::getDevs(platform);
    const auto& alpaka_host = alpaka::getDevByIdx(alpaka_common::PlatformHost(), 0u);
    CPPUNIT_ASSERT(alpaka_devices.size());
    const auto& alpaka_device = alpaka_devices[0];
    Queue queue{alpaka_device};

    const std::size_t batch_size = 32;

    // host structs
    PortableHostCollection<SoAInputs> inputs_host(batch_size, cms::alpakatools::host());
    PortableHostCollection<SoAOutputs> outputs_host(batch_size, cms::alpakatools::host());
    // device structs
    PortableCollection<SoAInputs, Device> inputs_device(batch_size, alpaka_device);
    PortableCollection<SoAOutputs, Device> outputs_device(batch_size, alpaka_device);

    // prepare inputs
    for (size_t i = 0; i < batch_size; i++) {
      inputs_host.view().x()[i] = 0.0f;
      inputs_host.view().y()[i] = 0.0f;
      inputs_host.view().z()[i] = 0.0f;
    }
    alpaka::memcpy(queue, inputs_device.buffer(), inputs_host.buffer());
    alpaka::wait(queue);

    {
      // guard scope
      cms::torch::alpaka::set_threading_guard();
      cms::torch::alpaka::Guard<Queue> guard(queue);

      // instantiate model
      auto m_path = get_path("/src/PhysicsTools/PyTorch/models/jit_classification_model.pt");
      auto model = cms::torch::alpaka::Model<cms::torch::alpaka::CompilationType::kJustInTime>(m_path);
      model.to(queue);
      CPPUNIT_ASSERT(cms::torch::alpaka::device(queue) == model.device());
      std::cout << "Device: " << model.device() << std::endl;

      // metadata for automatic tensor conversion
      auto input_records = inputs_device.view().records();
      auto output_records = outputs_device.view().records();
      cms::torch::alpaka::SoAMetadata<SoAInputs> inputs_metadata(batch_size);
      inputs_metadata.append_block("features", input_records.x(), input_records.y(), input_records.z());
      cms::torch::alpaka::SoAMetadata<SoAOutputs> outputs_metadata(batch_size);
      outputs_metadata.append_block("preds", output_records.m(), output_records.n());
      cms::torch::alpaka::ModelMetadata<SoAInputs, SoAOutputs> metadata(inputs_metadata, outputs_metadata);
      // inference
      model.forward(metadata);
      // check outputs
      alpaka::memcpy(queue, outputs_host.buffer(), outputs_device.buffer());
      alpaka::wait(queue);

      for (size_t i = 0; i < batch_size; i++) {
        CPPUNIT_ASSERT(outputs_host.const_view().m()[i] == 0.5f);
        CPPUNIT_ASSERT(outputs_host.const_view().n()[i] == 0.5f);
      }
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
