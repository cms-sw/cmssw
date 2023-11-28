/*
Based on https://github.com/Maverobot/libtorch_examples/blob/master/src/time_serie_prediction.cpp

Copyright (c) 2019, Zheng Qu
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
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
  // const size_t kOuputDim = 1;
  auto time_serie_detector = torch::nn::LSTM(
      torch::nn::LSTMOptions(kInputDim, kHiddenDim).dropout(0.2).num_layers(kSequenceLen).bidirectional(false));
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
