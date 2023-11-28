#include <torch/torch.h>

template <typename T>
void pretty_print(const std::string& info, T&& data) {
  std::cout << info << std::endl;
  std::cout << data << std::endl << std::endl;
}

int main(int /*argc*/, char* /*argv*/[]) {
  // Use GPU when present, CPU otherwise.
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA);
    std::cout << "CUDA is available! Training on GPU." << std::endl;
  }

  const size_t kSequenceLen = 1;
  const size_t kInputDim = 1;
  const size_t kHiddenDim = 5;
  const size_t kOuputDim = 1;
  auto time_serie_detector = torch::nn::LSTM(torch::nn::LSTMOptions(kInputDim, kHiddenDim)
                                                 .dropout(0.2)
                                                 .layers(kSequenceLen)
                                                 .bidirectional(false));
  time_serie_detector->to(device);
  std::cout << time_serie_detector << std::endl;

  torch::Tensor input = torch::empty({kSequenceLen, kInputDim});
  torch::Tensor state = torch::zeros({2, kSequenceLen, kHiddenDim});
  auto input_acc = input.accessor<float, 2>();
  size_t count = 0;
  for (float i = 0.1; i < 0.4; i += 0.1) {
    input_acc[count][0] = i;
    count++;
  }
  input = input.toBackend(c10::Backend::CUDA);
  state = state.toBackend(c10::Backend::CUDA);
  std::cout << "input = " << input << std::endl;
  time_serie_detector->zero_grad();

  auto i_tmp = input.view({input.size(0), 1, -1});
  auto s_tmp = state.view({2, state.size(0) / 2, 1, -1});

  pretty_print("input: ", i_tmp);
  pretty_print("state: ", s_tmp);

  auto rnn_output = time_serie_detector->forward(i_tmp, s_tmp);
  pretty_print("rnn_output/output: ", rnn_output.output);
  pretty_print("rnn_output/state: ", rnn_output.state);

  return 0;
}
