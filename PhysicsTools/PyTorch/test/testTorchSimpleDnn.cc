#include <torch/script.h>
#include "testBase.h"
#include <iostream>
#include <memory>
#include <vector>

class testSimpleDNN : public testBasePyTorch {
  CPPUNIT_TEST_SUITE(testSimpleDNN);
  CPPUNIT_TEST(test);
  CPPUNIT_TEST_SUITE_END();

public:
  std::string pyScript() const override;
  void test() override;
};

CPPUNIT_TEST_SUITE_REGISTRATION(testSimpleDNN);

std::string testSimpleDNN::pyScript() const { return "create_simple_dnn.py"; }

void testSimpleDNN::test() {
  std::string model_path = dataPath_ + "/simple_dnn.pt";
  torch::Device device(torch::kCPU);
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
