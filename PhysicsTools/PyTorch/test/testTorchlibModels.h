#ifndef PhysicsTools_PyTorch_test_testTorchlibModels_h
#define PhysicsTools_PyTorch_test_testTorchlibModels_h

#include <torch/torch.h>

namespace torchtest {

  class ClassifierModel : public torch::nn::Module {
  public:
    ClassifierModel(int in_dim = 3, int out_dim = 2) : fc(in_dim, out_dim), softmax(torch::nn::SoftmaxOptions(1)) {
      auto weight = torch::tensor({{1.0, 1.0, 0.0}, {1.0, 1.0, 0.0}}, torch::kFloat);
      fc->weight.set_data(weight);
      fc->weight.requires_grad_(false);

      auto bias = torch::tensor({0.0, 0.0}, torch::kFloat);
      fc->bias.set_data(bias);
      fc->bias.requires_grad_(false);

      register_module("fc", fc);
    }

    torch::Tensor forward(torch::Tensor x) {
      x = fc(x);
      x = softmax(x);
      return x;
    }

  private:
    torch::nn::Linear fc{nullptr};
    torch::nn::Softmax softmax;
  };

  class RegressionModel : public torch::nn::Module {
  public:
    RegressionModel(int in_dim = 3, int out_dim = 1) : fc(in_dim, out_dim) {
      fc->weight.set_data(torch::zeros_like(fc->weight));
      fc->weight.requires_grad_(false);

      fc->bias.set_data(torch::full_like(fc->bias, 0.5f));
      fc->bias.requires_grad_(false);

      fc = register_module("fc", fc);
    }

    torch::Tensor forward(torch::Tensor input) { return fc(input); }

  private:
    torch::nn::Linear fc{nullptr};
  };

  class LinearModel : public torch::nn::Module {
  public:
    LinearModel(int in_dim = 3, int out_dim = 2) : fc(in_dim, out_dim) {
      auto weights = torch::tensor({{-0.1f, 0.2f, 2.0f}, {0.1f, -2.3f, 4.0f}});
      fc->weight.set_data(weights);
      fc->weight.requires_grad_(false);

      fc->bias.set_data(torch::full_like(fc->bias, 0.0f));
      fc->bias.requires_grad_(false);

      fc = register_module("fc", fc);
    }

    torch::Tensor forward(torch::Tensor x) { return fc(x); }

  private:
    torch::nn::Linear fc{nullptr};
  };

  class MultiTaskModel : public torch::nn::Module {
  public:
    MultiTaskModel(int input_dim = 5)
        : fc1(input_dim, 128), fc2(128, 128), class_fc1(128, 64), class_fc2(64, 5), reg_fc1(128, 64), reg_fc2(64, 1) {
      fc1 = register_module("fc1", fc1);
      fc2 = register_module("fc2", fc2);
      class_fc1 = register_module("class_fc1", class_fc1);
      class_fc2 = register_module("class_fc2", class_fc2);
      reg_fc1 = register_module("reg_fc1", reg_fc1);
      reg_fc2 = register_module("reg_fc2", reg_fc2);

      // set weights and biases
      fc1->weight.set_data(torch::full_like(fc1->weight, 0.1f));
      fc1->bias.set_data(torch::full_like(fc1->bias, 0.0f));

      fc2->weight.set_data(torch::full_like(fc2->weight, 0.1f));
      fc2->bias.set_data(torch::full_like(fc2->bias, 0.0f));

      class_fc1->weight.set_data(torch::full_like(class_fc1->weight, 0.05f));
      class_fc1->bias.set_data(torch::full_like(class_fc1->bias, 0.0f));

      class_fc2->weight.set_data(torch::full_like(class_fc2->weight, 0.02f));
      class_fc2->bias.set_data(torch::full_like(class_fc2->bias, 0.0f));

      reg_fc1->weight.set_data(torch::full_like(reg_fc1->weight, 0.03f));
      reg_fc1->bias.set_data(torch::full_like(reg_fc1->bias, 0.0f));

      reg_fc2->weight.set_data(torch::full_like(reg_fc2->weight, 0.01f));
      reg_fc2->bias.set_data(torch::full_like(reg_fc2->bias, 0.0f));

      for (auto& p : this->parameters()) {
        p.set_requires_grad(false);
      }
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
      // Shared base
      x = fc1->forward(x);
      x = fc2->forward(x);

      // Classification head
      auto cls = class_fc1->forward(x);
      cls = class_fc2->forward(cls);
      cls = torch::softmax(cls, 1);

      // Regression head
      auto reg = reg_fc1->forward(x);
      reg = reg_fc2->forward(reg);

      return std::make_tuple(cls, reg);
    }

  private:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    torch::nn::Linear class_fc1{nullptr}, class_fc2{nullptr};
    torch::nn::Linear reg_fc1{nullptr}, reg_fc2{nullptr};
  };

}  // namespace torchtest

#endif  // PhysicsTools_PyTorch_test_testTorchlibModels_h
