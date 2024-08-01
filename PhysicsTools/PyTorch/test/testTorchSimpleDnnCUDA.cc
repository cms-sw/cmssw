#include <torch/script.h>
#include "testBaseCUDA.h"
#include <iostream>
#include <memory>
#include <vector>
#include "HeterogeneousCore/CUDAServices/interface/CUDAInterface.h"

class testSimpleDNNCUDA : public testBasePyTorchCUDA {
  CPPUNIT_TEST_SUITE(testSimpleDNNCUDA);
  CPPUNIT_TEST(test);
  CPPUNIT_TEST_SUITE_END();

public:
  std::string pyScript() const override;
  void test() override;
};

CPPUNIT_TEST_SUITE_REGISTRATION(testSimpleDNNCUDA);

std::string testSimpleDNNCUDA::pyScript() const { return "create_simple_dnn.py"; }

void testSimpleDNNCUDA::test() {
  std::vector<edm::ParameterSet> psets;
  edm::ServiceToken serviceToken = edm::ServiceRegistry::createSet(psets);
  edm::ServiceRegistry::Operate operate(serviceToken);

  // Setup the CUDA Service
  edmplugin::PluginManager::configure(edmplugin::standard::config());

  std::string const config = R"_(import FWCore.ParameterSet.Config as cms
process = cms.Process('Test')
process.add_(cms.Service('ResourceInformationService'))
process.add_(cms.Service('CUDAService'))
)_";
  std::unique_ptr<edm::ParameterSet> params;
  edm::makeParameterSets(config, params);
  edm::ServiceToken tempToken(edm::ServiceRegistry::createServicesFromConfig(std::move(params)));
  edm::ServiceRegistry::Operate operate2(tempToken);
  edm::Service<CUDAInterface> cuda;
  std::cout << "CUDA service enabled: " << cuda->enabled() << std::endl;

  std::cout << "Testing CUDA backend" << std::endl;

  std::string model_path = dataPath_ + "/simple_dnn.pt";
  torch::Device device(torch::kCUDA);
  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(model_path);
    module.to(device);
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n" << e.what() << std::endl;
    CPPUNIT_ASSERT(false);
  }
  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones(10, device));

  // Execute the model and turn its output into a tensor.
  at::Tensor output = module.forward(inputs).toTensor();
  std::cout << "output: " << output << '\n';
  CPPUNIT_ASSERT(output.item<float_t>() == 110.);
  std::cout << "ok\n";
}
