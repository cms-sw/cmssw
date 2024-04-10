#include <torch/torch.h>
#include <iostream>

struct Net : torch::nn::Module {
  Net(int64_t N, int64_t M) : linear(register_module("linear", torch::nn::Linear(N, M))) {
    another_bias = register_parameter("b", torch::randn(M));
  }
  torch::Tensor forward(torch::Tensor input) { return linear(input) + another_bias; }
  torch::nn::Linear linear;
  torch::Tensor another_bias;
};

int main(int /*argc*/, char* /*argv*/[]) {
  // Use GPU when present, CPU otherwise.
  Net net(4, 5);

  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA);
    std::cout << "CUDA is available! Training on GPU." << std::endl;
  }

  net.to(device);

  for (const auto& pair : net.named_parameters()) {
    std::cout << pair.key() << ": " << pair.value() << std::endl;
  }

  std::cout << net.forward(torch::ones({2, 4})) << std::endl;

  return 0;
}
