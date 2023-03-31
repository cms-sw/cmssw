#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAInterface.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "CUDATestDeviceAdditionAlgo.h"

class CUDATestDeviceAdditionModule : public edm::global::EDAnalyzer<> {
public:
  explicit CUDATestDeviceAdditionModule(edm::ParameterSet const& config);
  ~CUDATestDeviceAdditionModule() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void analyze(edm::StreamID, edm::Event const& event, edm::EventSetup const& setup) const override;

private:
  const uint32_t size_;
};

CUDATestDeviceAdditionModule::CUDATestDeviceAdditionModule(edm::ParameterSet const& config)
    : size_(config.getParameter<uint32_t>("size")) {}

void CUDATestDeviceAdditionModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<uint32_t>("size", 1024 * 1024);
  descriptions.addWithDefaultLabel(desc);
}

void CUDATestDeviceAdditionModule::analyze(edm::StreamID, edm::Event const& event, edm::EventSetup const& setup) const {
  // require CUDA for running
  edm::Service<CUDAInterface> cuda;
  if (not cuda or not cuda->enabled()) {
    std::cout << "The CUDAService is not available or disabled, the test will be skipped.\n";
    return;
  }

  // random number generator with a gaussian distribution
  std::random_device rd{};
  std::default_random_engine rand{rd()};
  std::normal_distribution<float> dist{0., 1.};

  // tolerance
  constexpr float epsilon = 0.000001;

  // allocate input and output host buffers
  std::vector<float> in1_h(size_);
  std::vector<float> in2_h(size_);
  std::vector<float> out_h(size_);

  // fill the input buffers with random data, and the output buffer with zeros
  for (size_t i = 0; i < size_; ++i) {
    in1_h[i] = dist(rand);
    in2_h[i] = dist(rand);
    out_h[i] = 0.;
  }

  // allocate input and output buffers on the device
  float* in1_d;
  float* in2_d;
  float* out_d;
  cudaCheck(cudaMalloc(&in1_d, size_ * sizeof(float)));
  cudaCheck(cudaMalloc(&in2_d, size_ * sizeof(float)));
  cudaCheck(cudaMalloc(&out_d, size_ * sizeof(float)));

  // copy the input data to the device
  cudaCheck(cudaMemcpy(in1_d, in1_h.data(), size_ * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(in2_d, in2_h.data(), size_ * sizeof(float), cudaMemcpyHostToDevice));

  // fill the output buffer with zeros
  cudaCheck(cudaMemset(out_d, 0, size_ * sizeof(float)));

  // launch the 1-dimensional kernel for vector addition
  HeterogeneousCoreCUDATestDevicePlugins::wrapper_add_vectors_f(in1_d, in2_d, out_d, size_);

  // copy the results from the device to the host
  cudaCheck(cudaMemcpy(out_h.data(), out_d, size_ * sizeof(float), cudaMemcpyDeviceToHost));

  // wait for all the operations to complete
  cudaCheck(cudaDeviceSynchronize());

  // check the results
  for (size_t i = 0; i < size_; ++i) {
    float sum = in1_h[i] + in2_h[i];
    assert(out_h[i] < sum + epsilon);
    assert(out_h[i] > sum - epsilon);
  }

  std::cout << "All tests passed.\n";
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CUDATestDeviceAdditionModule);
